from __future__ import annotations

import datetime

from evaluateur.prompts.queries import (
    QUERY_SYSTEM_TEMPLATE,
    format_query_prompts,
)
from evaluateur.queries.models import GeneratedTuple


def test_format_query_prompts_includes_today() -> None:
    t = GeneratedTuple({"topic": "billing"})
    sys_msg, _ = format_query_prompts(t, "some context")
    today = datetime.date.today().isoformat()
    assert today in sys_msg
    assert "time-sensitive" in sys_msg


def test_system_template_has_no_triple_blank_lines() -> None:
    assert "\n\n\n" not in QUERY_SYSTEM_TEMPLATE
