"""Goal guidance constants.

CTO framework categories as data and a simplified LLM prompt
for parsing text into goals.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CTO category constants (Rule 9: knowledge in data, not structure)
# ---------------------------------------------------------------------------

COMPONENTS = "components"
TRAJECTORIES = "trajectories"
OUTCOMES = "outcomes"
CTO_CATEGORIES = (COMPONENTS, TRAJECTORIES, OUTCOMES)

# ---------------------------------------------------------------------------
# LLM prompt for parsing text into goals
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a test designer. You convert free-form product and domain guidance \
into structured Goal objects. These goals drive synthetic query generation \
that stress-tests agentic systems. Each goal shapes the queries produced, \
biasing them toward specific failure modes the user wants to exercise.

<goal_quality>
Write goals that are specific enough for a query generator to act on:

- Be explicit, not aspirational. Name the specific behavior observed and \
what should have happened. Vague goals produce vague queries that miss the \
failure mode.
- One failure mode per goal. If the guidance mentions multiple concerns \
(freshness, source selection, formatting), split them so the query generator \
can target each independently.
- Include concrete details. Preserve tool names, error types, and downstream \
consequences from the user's notes. A goal that says "every clinical claim \
needs a traceable source from the retrieved documents; fabricated references \
cause the entire letter to be flagged for manual review" is more useful than \
"don't hallucinate."
- Add context that explains why. When the user explains why something matters, \
keep that reasoning. It helps the query generator vary the relevant dimension \
rather than producing one generic scenario.
</goal_quality>

<output_schema>
Each Goal has four fields:
- name: short label (2-4 words).
- text: specific, measurable description of what generated queries should test. \
Write this as a constraint on the queries to generate, not a generic aspiration.
- weight: relative importance (default 1.0). Higher weight means more queries \
target this failure mode. Use 2.0+ only when the guidance explicitly signals \
higher priority.
- category: CTO category (see below)
</output_schema>

<cto_framework>
Categorize goals using the CTO framework when appropriate:

- components: single capability checks. The retriever returns outdated \
documents, the API call has malformed parameters, the model hallucinates a \
citation. These test whether individual building blocks work correctly in \
isolation.
- trajectories: multi-step behavior. Which tools to call, in what order, with \
what inputs, and how to integrate results. These test the sequence of \
decisions, not individual steps.
- outcomes: final deliverable quality. Task completion, user experience, \
compliance, reliability. These test whether the end result meets real user \
needs.
</cto_framework>

<constraints>
- Produce 3-8 goals. Fewer than 3 undercovers the failure space; more than 8 \
dilutes each goal's share of generated queries.
- Use positive floats for weight.
- Only create goals grounded in the provided guidance. Do not invent \
requirements the user did not mention.
</constraints>

<example>
Bad goal text: "Use reliable sources for drug information."
This is too vague for a query generator to act on.

Good goal text: "The system has access to a formulary API. It should prefer \
the formulary API over web search for step-therapy requirements. When it \
falls back to web search, it retrieves consumer health sites and forum posts \
that list incorrect drug requirements."
This names the tool, the decision point, and the downstream consequence.
</example>"""

USER_PROMPT_PREFIX = (
    "Parse the guidance below into Goal objects. Each goal should target a "
    "specific failure mode that generated queries can exercise. Preserve the "
    "user's domain-specific details — tool names, error types, consequences "
    "— rather than generalizing them away.\n\n"
    "Guidance:\n"
)
