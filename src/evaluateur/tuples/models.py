from __future__ import annotations

from collections.abc import ItemsView, Iterator, KeysView

from pydantic import RootModel

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
        fields = ", ".join(f"{k}={v!r}" for k, v in self.root.items())
        return f"GeneratedTuple({fields})"

    def __rich_repr__(self) -> Iterator[tuple[str, ScalarValue]]:
        yield from self.root.items()
