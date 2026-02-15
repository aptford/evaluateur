from __future__ import annotations

from collections.abc import ItemsView, KeysView

from pydantic import BaseModel, ConfigDict, Field, RootModel

from evaluateur.goals.models import GoalMode, GoalSpec
from evaluateur.options.types import ScalarValue


class GeneratedTuple(RootModel[dict[str, ScalarValue]]):
    """A concrete combination of dimension values.

    Dimension key-value pairs are stored directly (no wrapper).
    Supports dict-like access::

        t["payer"]              # item access
        t.get("payer", "n/a")   # safe access with default
        t.items()               # iterate key-value pairs
        "payer" in t            # membership test
    """

    def __getitem__(self, key: str) -> ScalarValue:
        return self.root[key]

    def __contains__(self, key: object) -> bool:
        return key in self.root

    def get(self, key: str, default: ScalarValue = None) -> ScalarValue:
        return self.root.get(key, default)

    def items(self) -> ItemsView[str, ScalarValue]:
        return self.root.items()

    def keys(self) -> KeysView[str]:
        return self.root.keys()

    def __len__(self) -> int:
        return len(self.root)

    def __bool__(self) -> bool:
        return bool(self.root)

    def __repr__(self) -> str:
        return f"GeneratedTuple({self.root!r})"


class QueryMetadata(BaseModel):
    """Metadata associated with a generated query.

    This includes both:
    - run-level metadata injected by the evaluator (mode, goal_guided, query_goals)
    - per-query metadata set by generators (free-form keys)

    Extra keys are allowed for experimentation and backend-specific tracing.
    """

    model_config = ConfigDict(extra="allow")

    goal_guided: bool = False
    query_goals: GoalSpec | None = None
    goal_mode: GoalMode | None = None
    goal_focus: str | None = None
    goal_category: str | None = None


class GeneratedQuery(BaseModel):
    """Natural language query with full traceability back to its tuple."""

    query: str
    source_tuple: GeneratedTuple
    metadata: QueryMetadata = Field(default_factory=QueryMetadata)
