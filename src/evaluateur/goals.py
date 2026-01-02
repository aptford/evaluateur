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
            "You convert free-form user guidance into a structured GoalSpec used to generate "
            "synthetic evaluation queries.\n\n"
            "Framework (three layers to target):\n"
            "- Components: individual building blocks (e.g. retrieval, tool calls, extraction, "
            "grounding/citations). These are unit-test style checks.\n"
            "- Trajectories: the sequence of decisions and recovery behavior (e.g. tool choice, order, "
            "retries, detecting staleness/missing info, handling conflicts, escalation).\n"
            "- Outcomes: what ships to users (task completion, checklist compliance, UX constraints, "
            "reliability under change).\n\n"
            "Critical constraint: this GoalSpec conditions QUERY GENERATION. The goals should be "
            "written so they can guide a query generator to produce queries that stress-test the "
            "target system.\n\n"
            "Primary output preference:\n"
            "- Prefer rich, explicit GoalItem.description over must_include/avoid.\n"
            "- Populate GoalItem.examples with concrete example user queries.\n\n"
            "Mapping rules (be concrete):\n"
            "- Each layer should contain a small number of GoalItems (typically 1-5).\n"
            "- For each GoalItem:\n"
            "  - name: short label.\n"
            "  - description: 2-5 sentences describing what the generated user query should stress, "
            "what failure mode it targets, and what a good response behavior would look like.\n"
            "  - examples: 1-3 fully-formed, natural-language example user queries that would "
            "satisfy this goal.\n"
            "  - must_include/avoid: use only when the user explicitly asks for specific phrases or "
            "when a literal checklist token is essential; otherwise leave them empty.\n"
            "- Put assumptions / ambiguity handling in the layer summary.\n\n"
            "What good goals look like (test-oriented):\n"
            "- Freshness/staleness: force checking effective dates / most recent versions.\n"
            '- Coverage gaps: force detecting missing required sources and saying "not found".\n'
            "- Tool semantics: force preferring authoritative sources/tools over generic web search.\n"
            "- Conflicts: force detecting conflicting evidence and reconciling or escalating.\n"
            "- Escalation/uncertainty: force asking for confirmation or flagging uncertainty.\n"
            "- Outcome constraints: force checklist-ready, workflow-usable outputs.\n\n"
            "Output constraints (GPT-5.2 best practices):\n"
            '- Be specific and test-oriented; avoid vague statements like "be high quality".\n'
            "- If guidance is broad, infer reasonable goals across ALL THREE layers.\n"
            "- Use weight (default 1.0) to reflect priority; set 0.0 to disable goals.\n"
            "- Do not invent domain facts; do not introduce new requirements unrelated to the input.\n"
        )

        user = (
            "Turn the following guidance into a GoalSpec.\n\n"
            "Important:\n"
            "- Prefer verbose GoalItem.description.\n"
            "- Include 1-3 GoalItem.examples per goal (example user queries).\n"
            "- Use must_include/avoid only if explicitly requested or clearly necessary.\n"
            "- If the guidance is broad, infer actionable goals in components, trajectories, and "
            "outcomes.\n\n"
            "User guidance:\n"
            f"{cleaned}"
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
