from __future__ import annotations

import textwrap
from typing import Any, Iterable

from pydantic import BaseModel, Field

from evaluateur.client import LLMClient


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


class GoalLayer(BaseModel):
    """A set of goals for one framework layer (components/trajectories/outcomes)."""

    summary: str | None = Field(
        default=None,
        description="Optional one-paragraph summary of what matters in this layer.",
    )
    items: list[GoalItem] = Field(default_factory=list)


class GoalSpec(BaseModel):
    """User-provided guidance for shaping evaluation queries.

    This mirrors the Components / Trajectories / Outcomes framework.

    The spec is used in two ways:

    1) **Per-run prompt conditioning**: we render it to a short instruction block
       and attach it to the query-generation context.
    2) **Compile-time optimization**: a DSPy teleprompter can use it as a rubric
       to score candidate prompts/few-shot selections.
    """

    title: str | None = Field(default=None, description="Optional name for this spec")

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
            "You convert user intent into a structured goal specification for generating "
            "synthetic evaluation queries. Extract goals across three layers: components, "
            "trajectories, outcomes.\n\n"
            "Rules:\n"
            "- Keep goals actionable and test-oriented (what should a query force the system to do?).\n"
            "- Prefer short 'must_include' checklist tokens when possible.\n"
            "- Use weights to represent relative priority (default 1.0).\n"
            "- If the user is ambiguous, make reasonable assumptions and keep them in 'summary'.\n"
        )

        user = (
            "Turn the following user guidance into a GoalSpec.\n\n"
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
            and not (self.title or self.components.summary or self.trajectories.summary or self.outcomes.summary)
        )

    def render_prompt(self, *, max_chars: int = 1800) -> str:
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
                    extra.append("must include: " + ", ".join(it.must_include))
                if it.avoid:
                    extra.append("avoid: " + ", ".join(it.avoid))
                if extra:
                    line += " (" + "; ".join(extra) + ")"
                lines.append(f"- {line}")
            return lines

        chunks: list[str] = []
        header = "Query optimization goals"
        if self.title:
            header += f" — {self.title}"
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
        if len(rendered) <= max_chars:
            return rendered

        # If too long, drop examples implicitly (we never render them) and truncate.
        return rendered[: max_chars - 1].rstrip() + "…\n"

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable metadata representation."""

        return self.model_dump(exclude_none=True)
