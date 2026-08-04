"""The policy agent's tool: search the policy docs, return citable excerpts.

WHY A TOOL LAYER ON TOP OF THE RETRIEVER (instead of the agent calling
rag/retriever.py directly)
    Same split as order_tools over db/pool: the retriever is an ENGINE
    (query in, Chunk objects out - reusable by evals, by the output rail,
    by anything); the tool is the PRESENTATION of that engine to a model -
    docstring that earns correct usage, arguments a model can't fumble,
    results formatted as a prompt. Engines don't talk to models; tools do.

THE ONE BIG IDEA IN THIS FILE: THE CITATION CONTRACT RIDES WITH THE DATA
    The tool result doesn't just contain the policy text - its FIRST LINE
    instructs the model to cite section ids for every claim it makes.
    Why put the instruction here instead of (only) the system prompt?
    Recency: by the time the model composes its answer, the system prompt
    is thousands of tokens back, but the tool result was the LAST thing
    it read. Instructions work best delivered at the moment of use.
    (The output rail in Phase 7 then CHECKS the contract from the other
    side: policy claims without citations get flagged. Promise here,
    enforcement there.)

Run directly - no database, no LLM, no key needed:

    python -m tools.policy_tools
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from rag.retriever import get_retriever


class SearchPoliciesArgs(BaseModel):
    query: str = Field(
        min_length=3,
        max_length=200,
        description=(
            "The customer's policy question, mostly in their own words, "
            "e.g. 'how long do I have to return headphones?'. Do not "
            "compress it to lone keywords; the search handles sentences."
        ),
    )
    # Note what is NOT here: no `k`, no source-file filter, no score
    # threshold. Every knob exposed to the model is a knob the model can
    # turn wrongly. Tuning parameters belong to us (and later, to evals) -
    # the model just asks questions.


@tool("search_policies", args_schema=SearchPoliciesArgs)
def search_policies(query: str) -> str:
    """Search the official store policies (returns, shipping, warranty).

    WHAT: Returns the most relevant policy sections, verbatim, each with a
    citation id like [RET-1].

    USE WHEN: The customer asks what is allowed, covered, free, refundable,
    cancellable, or how long something takes as a matter of POLICY - e.g.
    'can I return this?', 'is shipping free?', 'is my warranty still valid?'.

    DO NOT USE: For facts about a SPECIFIC order (status, tracking, dates -
    use get_order_status) or about a customer's account (find_customer).
    Policy says what the rules are; the order tools say what actually
    happened.

    PREREQUISITES: None. Call it before answering ANY policy question -
    never answer policy from memory, even when confident.

    RETURNS: Up to 3 policy excerpts, each headed by its citation id, with
    citation instructions on the first line. If nothing relevant exists:
    'NO POLICY FOUND ...' - in that case say the topic is not covered by
    the published policies and offer to connect the customer with human
    support. NEVER invent policy.
    """
    try:
        hits = get_retriever().search(query, k=3)
    except RuntimeError as e:
        # Index failed to build (docs missing/malformed) - a developer
        # problem, reported honestly so the model doesn't improvise rules.
        return (
            f"CONFIGURATION ERROR: the policy index is unavailable ({e}). "
            "Tell the customer you cannot check policies right now and "
            "offer human support. Do not answer from memory."
        )

    if not hits:
        return (
            f"NO POLICY FOUND matching '{query}'. The published policies "
            "cover returns, shipping, and warranty only. Tell the customer "
            "this topic is not covered by the published policies and offer "
            "to connect them with human support. Do NOT invent or guess "
            "policy."
        )

    # The contract line, then the excerpts. Each excerpt is VERBATIM
    # policy text - we never summarise here. Summarising is the model's
    # job, with the original in front of it; a paraphrase in the tool
    # result would put OUR mistakes into ITS citations.
    lines = [
        "POLICY EXCERPTS - when you use a fact from an excerpt, cite its "
        "section id (e.g. [RET-1]) right next to that fact. Do not state "
        "policy facts that are not supported by an excerpt below.",
    ]
    for c in hits:
        lines.append("")
        lines.append(f"--- {c.label()} ---")
        lines.append(c.text)
    return "\n".join(lines)


POLICY_TOOLS = [search_policies]


# ---------------------------------------------------------------------------
# Smoke test - files and math only.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for q in [
        "how long do I have to return something?",
        "my headphones broke after two months, am I covered?",
        "do you deliver for free?",
        "can I pay with cryptocurrency?",   # -> NO POLICY FOUND path
    ]:
        print(f"\n================ query: {q}")
        print(search_policies.invoke({"query": q}))
