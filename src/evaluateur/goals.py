from __future__ import annotations

import textwrap
import math
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field, field_validator

from evaluateur.client import LLMClient


GoalFocusArea = Literal["components", "trajectories", "outcomes"]
GoalMode = Literal["full", "sample"]


class GoalItem(BaseModel):
    """A single user goal used to guide query optimization.

    Goals are intentionally flexible: they can represent checklists (binary),
    weighted preferences, or concrete inclusion/avoidance constraints.
    """

    name: str = Field(..., description="Short goal name")
    description: str | None = Field(
        default=None, description="Plain-language description of the goal"
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Relative importance. 0 disables the goal without deleting it.",
    )

    must_include: list[str] = Field(
        default_factory=list,
        description=(
            "Tokens/phrases/requirements the query should explicitly include. "
            "Use for checklist-style constraints."
        ),
    )
    avoid: list[str] = Field(
        default_factory=list,
        description="Tokens/phrases/requirements the query should avoid.",
    )
    examples: list[str] = Field(
        default_factory=list,
        description="Optional examples of queries that satisfy this goal.",
    )

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, v: float) -> float:
        """Ensure weight is a finite, non-negative float.

        Note: 0.0 is allowed and used as a "disabled goal" sentinel.
        """

        # Pydantic will typically coerce int/str inputs to float before validators.
        # Keep this explicit to ensure consistent output type.
        v = float(v)
        if not math.isfinite(v):
            raise ValueError("weight must be a finite number")
        if v < 0:
            raise ValueError("weight must be >= 0")
        return v


class GoalLayer(BaseModel):
    """A set of goals for one framework layer (components/trajectories/outcomes)."""

    summary: str | None = Field(
        default=None,
        description="Optional one-paragraph summary of what matters in this layer.",
    )
    items: list[GoalItem] = Field(default_factory=list)


class GoalSpec(BaseModel):
    """User-provided guidance for shaping evaluation queries."""

    components: GoalLayer = Field(default_factory=GoalLayer)
    trajectories: GoalLayer = Field(default_factory=GoalLayer)
    outcomes: GoalLayer = Field(default_factory=GoalLayer)

    @classmethod
    async def from_text(cls, client: LLMClient, text: str) -> GoalSpec:
        """Parse free-form user text into a structured `GoalSpec`.

        This uses Instructor + Pydantic parsing, so callers get a stable schema.
        """

        cleaned = text.strip()
        if not cleaned:
            return cls()

        system = (
            "You are a test designer. Convert free-form product/domain guidance into a structured "
            "GoalSpec that will be used to GENERATE synthetic evaluation user queries.\n\n"
            "Your output is consumed by a query generator. Write goals as measurable constraints "
            "on the *user queries to generate* (not as generic product requirements).\n\n"
            "Assume the target is an agentic system by default.\n"
            "First, infer a plausible 'system under test' (SUT) component inventory from the guidance. "
            "Infer the kinds of components an agentic product usually "
            "has and what would be high-risk in the given domain.\n\n"
            "Typical agentic components to consider (pick only those relevant):\n"
            "- Input understanding: intent classification, entity extraction, constraint parsing.\n"
            "- Knowledge access: retrieval/search, policy lookup, tool/API calls, database queries.\n"
            "- Tool routing: choosing the right tool, parameterization, error handling.\n"
            "- Reasoning/control: planning, step ordering, branching, stopping conditions.\n"
            "- State: memory, session context, preference handling, caching.\n"
            "- Safety/compliance: refusals, escalation, sensitive data handling, policy adherence.\n"
            "- Output shaping: structured outputs, formatting, citations/grounding, UX constraints.\n\n"
            "Framework (three layers; MUST be non-overlapping):\n"
            "- Components: single capability checks (unit-test style). One primary metric family per goal.\n"
            "- Trajectories: multi-step behavior over time (decision sequence, recovery, retries, escalation).\n"
            "- Outcomes: what the user receives (deliverable quality, completeness, compliance, usability).\n\n"
            "Non-overlap contract:\n"
            "- Do not duplicate the same failure mode or metric family across layers.\n"
            "- Components describe 'can it do X once?'; trajectories describe 'does it take the right path?'; "
            "outcomes describe 'is the final artifact usable/compliant?'\n\n"
            "Metric menu (choose a few that match the domain; make them measurable):\n"
            "- Tool choice: uses the most authoritative available tool/source (vs generic search) for the need.\n"
            '- Tool call semantics: correct parameters; detects semantic failures (e.g. "not found") and recovers.\n'
            "- Retrieval sufficiency: covers all required evidence categories for the task (not just relevance@k).\n"
            "- Freshness/staleness: checks effective dates/'as of'; prefers newest policy/version; flags uncertainty.\n"
            '- Coverage-gap handling: explicitly asks for missing required inputs or returns "not found".\n'
            "- Conflict integration: detects contradictions; applies precedence rules; escalates when unresolved.\n"
            "- Faithfulness/grounding: every key claim is attributable to retrieved/tool outputs; no fabricated cites.\n"
            "- State/memory correctness: uses prior turns/records when appropriate; avoids stale carryover.\n"
            "- Safety/compliance: follows policy/PII constraints; refuses/escalates unsafe or disallowed requests.\n"
            "- Efficiency/cost: minimizes redundant steps/tool calls; bounded retries; latency/token budget awareness.\n\n"
            "Precision requirement: Every goal MUST include quantifiable pass criteria.\n"
            "Because the schema has no dedicated metric fields, embed metrics inside GoalItem.description using "
            "this STRICT template (use these exact labels):\n"
            "```\n"
            "Target behavior: ...\n"
            "Failure mode: ...\n"
            "Observable signals: ...\n"
            "Pass criteria: ... (use thresholds like counts, required fields, ordering constraints, "
            "or explicit binary checks)\n"
            "Stress knobs: ... (how to make queries harder)\n"
            "```\n\n"
            'Anti-vagueness rule: Do not write goals like "be accurate" or "high quality". Replace them '
            "with observable checks (counts, required sections, explicit decision points, ordering constraints, "
            "or binary conditions).\n\n"
            "Examples requirement: Each GoalItem MUST include 2-3 example user queries that would satisfy "
            "the goal. Examples should sound like real users in the implied domain, include realistic constraints "
            "(time, jurisdiction, policy, budget, risk), and include an explicit acceptance sentence such as "
            '"Please cite...", "If you can\'t find..., say not found", or "Return JSON with keys ...".\n\n'
            "Example non-overlap requirement (STRICT):\n"
            "- No example user query may appear in more than one layer (components vs trajectories vs outcomes).\n"
            "- Within a layer, do not reuse the same underlying scenario across different GoalItems.\n"
            "- Make examples distinct by task type, constraints, and the primary metric being stressed.\n"
            "- Enforce uniqueness by assigning each example a hidden scenario ID and ensuring each ID is used "
            "exactly once across the entire GoalSpec.\n\n"
            "Mapping rules:\n"
            "- Each layer should contain 2-4 GoalItems (never more than 5).\n"
            "- Use a positive float as weight to reflect priority (default 1.0).\n"
            "- Prefer rich GoalItem.description; avoid must_include/avoid unless the user explicitly requests "
            "literal tokens/phrases.\n"
            "- Put assumptions, ambiguity handling, and inferred SUT components in the layer summary.\n"
            "- Do not add new requirements unrelated to the provided guidance.\n"
        )

        user = (
            "Create a GoalSpec from the guidance below.\n\n"
            "Interpret the guidance as describing:\n"
            "- who the users are\n"
            "- what tasks they do\n"
            "- what domain constraints exist (policy, compliance, safety, correctness)\n"
            "- what kinds of tools/knowledge sources the SUT likely uses\n\n"
            "Output requirements recap:\n"
            "- 2-4 goals per layer, non-overlapping.\n"
            "- Every GoalItem.description MUST follow the labeled metric template:\n"
            "  Target behavior / Failure mode / Observable signals / Pass criteria / Stress knobs.\n"
            "- Every GoalItem.examples: 2-3 natural user queries with realistic constraints and an "
            "explicit acceptance sentence. Make the voice genuinely authentic for the domain (role, "
            "jargon, workflow pressures), without inventing external facts.\n"
            "- Keep must_include/avoid empty unless a literal checklist token is essential.\n\n"
            "Guidance:\n"
            f"```{cleaned}```"
        )

        inst = client.instructor_client
        parsed: GoalSpec = await inst.chat.completions.create(
            model=client.model_name,
            response_model=cls,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return parsed

    def is_empty(self) -> bool:
        """Return True if no goals are specified."""

        return (
            not self.components.items
            and not self.trajectories.items
            and not self.outcomes.items
            and not (
                self.components.summary
                or self.trajectories.summary
                or self.outcomes.summary
            )
        )

    def render_prompt(self, *, max_chars: int | None = None) -> str:
        """Render this spec into a compact instruction block.

        The output is designed to be appended to the evaluator's domain context.
        """

        def _render_items(items: Iterable[GoalItem]) -> list[str]:
            lines: list[str] = []
            for it in items:
                if it.weight <= 0:
                    continue
                parts: list[str] = [it.name]
                if it.description:
                    parts.append(it.description)

                line = " - ".join(parts)
                extra: list[str] = []
                if it.must_include:
                    extra.append(
                        "must include: "
                        + ", ".join(f"`{it}`" for it in it.must_include)
                    )
                if it.avoid:
                    extra.append("avoid: " + ", ".join(f"`{it}`" for it in it.avoid))
                if extra:
                    line += " (" + "; ".join(extra) + ")"
                lines.append(f"- {line}")

                if it.examples:
                    examples = [ex.strip() for ex in it.examples if ex.strip()]
                    if examples:
                        lines.append("  - examples:")
                        lines.extend(f'    - "{ex}"' for ex in examples)
            return lines

        chunks: list[str] = []
        header = "Query optimization goals"
        chunks.append(header + ":")

        if self.components.summary or self.components.items:
            chunks.append("\nComponents:")
            if self.components.summary:
                chunks.append(textwrap.fill(self.components.summary, width=96))
            chunks.extend(_render_items(self.components.items))

        if self.trajectories.summary or self.trajectories.items:
            chunks.append("\nTrajectories:")
            if self.trajectories.summary:
                chunks.append(textwrap.fill(self.trajectories.summary, width=96))
            chunks.extend(_render_items(self.trajectories.items))

        if self.outcomes.summary or self.outcomes.items:
            chunks.append("\nOutcomes:")
            if self.outcomes.summary:
                chunks.append(textwrap.fill(self.outcomes.summary, width=96))
            chunks.extend(_render_items(self.outcomes.items))

        rendered = "\n".join(chunks).strip() + "\n"
        if not isinstance(max_chars, int):
            return rendered
        if len(rendered) <= max_chars:
            return rendered

        suffix = "…\n"
        if max_chars <= 0:
            return ""
        if max_chars <= len(suffix):
            return suffix[:max_chars]
        cut = max_chars - len(suffix)
        return rendered[:cut].rstrip() + suffix

    def available_focus_areas(self) -> list[GoalFocusArea]:
        """Return the goal layers that are non-empty (considering weights)."""

        def _has_any(layer: GoalLayer) -> bool:
            if layer.summary and layer.summary.strip():
                return True
            return any(it.weight > 0 for it in layer.items)

        areas: list[GoalFocusArea] = []
        if _has_any(self.components):
            areas.append("components")
        if _has_any(self.trajectories):
            areas.append("trajectories")
        if _has_any(self.outcomes):
            areas.append("outcomes")
        return areas

    def render_focused_prompt(
        self, *, focus_area: GoalFocusArea, max_chars: int | None = None
    ) -> str:
        """Render only a single goal layer (components/trajectories/outcomes).

        Intended for sampling mode, where each generated query is conditioned on a
        single focus area to increase diversity.
        """

        layer: GoalLayer
        label: str
        if focus_area == "components":
            layer = self.components
            label = "Components"
        elif focus_area == "trajectories":
            layer = self.trajectories
            label = "Trajectories"
        else:
            layer = self.outcomes
            label = "Outcomes"

        # Reuse the same formatting conventions as render_prompt(), but restrict
        # to a single section.
        def _render_items(items: Iterable[GoalItem]) -> list[str]:
            lines: list[str] = []
            for it in items:
                if it.weight <= 0:
                    continue
                parts: list[str] = [it.name]
                if it.description:
                    parts.append(it.description)

                line = " - ".join(parts)
                extra: list[str] = []
                if it.must_include:
                    extra.append(
                        "must include: "
                        + ", ".join(f"`{it}`" for it in it.must_include)
                    )
                if it.avoid:
                    extra.append("avoid: " + ", ".join(f"`{it}`" for it in it.avoid))
                if extra:
                    line += " (" + "; ".join(extra) + ")"
                lines.append(f"- {line}")

                if it.examples:
                    examples = [ex.strip() for ex in it.examples if ex.strip()]
                    if examples:
                        lines.append("  - examples:")
                        lines.extend(f'    - "{ex}"' for ex in examples)
            return lines

        chunks: list[str] = []
        chunks.append(f"Query optimization goals (focus area: {label}):")
        chunks.append(f"\n{label}:")
        if layer.summary:
            chunks.append(textwrap.fill(layer.summary, width=96))
        chunks.extend(_render_items(layer.items))

        rendered = "\n".join(chunks).strip() + "\n"
        if not isinstance(max_chars, int):
            return rendered
        if len(rendered) <= max_chars:
            return rendered

        suffix = "…\n"
        if max_chars <= 0:
            return ""
        if max_chars <= len(suffix):
            return suffix[:max_chars]
        cut = max_chars - len(suffix)
        return rendered[:cut].rstrip() + suffix

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable metadata representation."""

        return self.model_dump(exclude_none=True)
