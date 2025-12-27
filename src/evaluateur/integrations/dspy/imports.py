from __future__ import annotations

from typing import Any


def import_dspy() -> Any | None:
    """Import DSPy if available, otherwise return None.

    This is the only place Evaluateur imports DSPy directly. All other modules
    should call this (or `require_dspy`) to keep DSPy optional and to avoid
    import-time side effects.
    """

    try:
        import dspy  # type: ignore[import]
    except Exception:  # pragma: no cover - optional dependency
        return None
    return dspy


def require_dspy() -> Any:
    """Return the DSPy module, raising a clear error if not installed."""

    dspy = import_dspy()
    if dspy is None:  # pragma: no cover
        raise RuntimeError("DSPy is not installed but DSPy query mode was requested.")
    return dspy


