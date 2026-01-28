from __future__ import annotations

from typing import TypedDict


class _Message(TypedDict):
    role: str
    content: str


_SYSTEM_PROMPT = (
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

_USER_PROMPT_PREFIX = (
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
)


def build_goal_spec_messages(text: str) -> list[_Message]:
    """Build instructor messages for parsing free-form goal guidance."""
    cleaned = text.strip()
    if not cleaned:
        return []
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"{_USER_PROMPT_PREFIX}```{cleaned}```"},
    ]
