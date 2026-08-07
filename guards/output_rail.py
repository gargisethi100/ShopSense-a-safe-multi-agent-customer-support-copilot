"""The output rail: check what leaves, after the agents have finished.

WHY GUARD BOTH ENDS
    The input gate stops bad things ARRIVING. The output rail stops bad
    things LEAVING - and they are different failures. A perfectly innocent
    question can still produce an answer that leaks a card number from a
    tool result, or states a policy the documents never contained. No
    input check can catch either, because neither exists yet at input time.

TWO CHECKS, BOTH FREE
    1. PII SWEEP        card / national-id patterns in the outgoing text
    2. CITATION AUDIT   the promise made in Phase 3, finally enforced

THE CITATION AUDIT IS THE INTERESTING ONE
    tools/policy_tools.py opens every result with "cite the section id for
    every fact". That is a PROMISE the model makes. This file is where the
    promise gets CHECKED, from the other side, by code that cannot be
    talked out of it. Two distinct failures:

      MISSING   policy excerpts were retrieved, but the answer cites none.
                The model may have answered from memory - exactly what the
                whole RAG apparatus exists to prevent.
      INVENTED  the answer cites [RET-9], which does not exist. A fabricated
                citation is worse than none: it looks verifiable and is not.

    INVENTED is checkable with certainty (compare against the corpus), so
    it is treated as a hard failure even in monitor mode - a citation to a
    nonexistent section is never a false positive.

Run directly (no model, no database - pure text checks):

    python -m guards.output_rail
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, ToolMessage

from config import get_settings
from graph.state import ShopSenseState
from guards.input_gate import _log_trigger, mask_for_log

# Citations look like [RET-1]. Same shape the docs, the retriever, and the
# policy tool all agree on - one convention, checkable from any side.
_CITATION = re.compile(r"\[([A-Z]{2,5}-\d+)\]")

# Marker that policy excerpts were actually put in front of the model.
_EXCERPT_HEADER = re.compile(r"^--- \[([A-Z]{2,5}-\d+)\]", re.MULTILINE)

# Outbound PII. Deliberately narrower than the input gate's list: the
# customer's own email and phone are legitimate in an answer ("I'll email
# dana@example.com"), but a card number never is - not from a tool result,
# not from the transcript, not ever.
_LEAK_PATTERNS = [
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[CARD]", "card-number"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]", "national-id"),
]


def valid_section_ids() -> set[str]:
    """Every citation id the corpus actually contains."""
    from rag.retriever import get_retriever

    return {c.section_id for c in get_retriever().chunks}


def sweep_pii(text: str) -> tuple[str, list[str]]:
    """Redact anything that must never reach a customer."""
    hits: list[str] = []
    for pattern, token, label in _LEAK_PATTERNS:
        text, n = pattern.subn(token, text)
        if n:
            hits.append(label)
    return text, hits


def audit_citations(answer: str, excerpts_shown: set[str]) -> list[str]:
    """Compare what the answer claims against what it was actually given."""
    problems: list[str] = []
    cited = set(_CITATION.findall(answer))

    if invented := cited - valid_section_ids():
        problems.append(f"invented-citation:{','.join(sorted(invented))}")

    if excerpts_shown and not cited:
        # Excerpts were retrieved and the answer cites nothing. Note the
        # deliberate narrowness: we do NOT flag an uncited answer when no
        # excerpts were retrieved, because "the policies don't cover that"
        # is a correct, citation-free answer. Flagging it would train the
        # team to ignore the alarm.
        problems.append("uncited-policy-answer")

    if unshown := cited - excerpts_shown - {"", *invented}:
        # Cited a REAL section that was never retrieved this turn. Usually
        # recall from earlier in the conversation - suspicious, not wrong.
        problems.append(f"cited-without-retrieval:{','.join(sorted(unshown))}")

    return problems


def output_rail_node(state: ShopSenseState) -> dict:
    """Inspect the final answer. Returns a corrected message if needed."""
    settings = get_settings()
    enforcing = settings.guardrails_mode == "enforce"
    messages = state.get("messages") or []

    answer = next(
        (m for m in reversed(messages)
         if isinstance(m, AIMessage) and not m.tool_calls),
        None,
    )
    if answer is None:
        return {}

    text = answer.text
    flags: list[str] = []
    updates: dict = {}

    # --- 1. PII sweep -----------------------------------------------------
    # Applied in BOTH modes, like input masking: a leaked card number is
    # not a tuning question. `monitor` exists to calibrate detection, never
    # to knowingly ship a leak.
    cleaned, leaks = sweep_pii(text)
    if leaks:
        _log_trigger("output-pii", ",".join(leaks), text, "masked")
        flags.append(f"output_rail: masked {','.join(leaks)} in answer")
        updates["messages"] = [AIMessage(content=cleaned, id=answer.id)]

    # --- 2. citation audit ------------------------------------------------
    # What was actually put in front of the model this turn?
    shown: set[str] = set()
    for m in messages:
        if isinstance(m, ToolMessage):
            shown |= set(_EXCERPT_HEADER.findall(str(m.content)))

    if problems := audit_citations(cleaned, shown):
        hard = any(p.startswith("invented-citation") for p in problems)
        action = "flagged"
        _log_trigger("citation", ";".join(problems), text, action)
        flags.extend(f"output_rail: {p}" for p in problems)

        if hard and enforcing:
            # An invented citation is a verifiable fabrication, so this is
            # the one output problem we correct rather than merely record.
            updates["messages"] = [AIMessage(
                content=(
                    "I need to double-check that against our published "
                    "policies before I answer - I don't want to quote a "
                    "rule I can't point to. Could you give me a moment, or "
                    "shall I connect you with a human colleague?"
                ),
                id=answer.id,
            )]

    return {**updates, **({"gate_flags": flags} if flags else {})}


# ---------------------------------------------------------------------------
# Smoke test - pure text, no model, no database.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    from config import enable_utf8_console

    enable_utf8_console()

    print(f"valid section ids in corpus: {len(valid_section_ids())}")
    print(f"mode: {get_settings().guardrails_mode}\n")

    def excerpt(*ids: str) -> ToolMessage:
        body = "\n".join(f"--- [{i}] Some title (returns.md) ---\ntext" for i in ids)
        return ToolMessage(content=body, tool_call_id="t1")

    CASES = [
        (
            "good: cited, and the citation was retrieved",
            [excerpt("RET-1"), AIMessage(content="You have 30 days [RET-1].")],
        ),
        (
            "MISSING: excerpts retrieved, answer cites nothing",
            [excerpt("RET-1"), AIMessage(content="You have 30 days to return it.")],
        ),
        (
            "INVENTED: cites a section that does not exist",
            [excerpt("RET-1"), AIMessage(content="Returns are free forever [RET-99].")],
        ),
        (
            "ok: no excerpts, no citations - a correct refusal",
            [AIMessage(content="Our published policies don't cover that.")],
        ),
        (
            "LEAK: a card number reached the answer",
            [AIMessage(content="I refunded card 4111 1111 1111 1111 today.")],
        ),
    ]

    for label, msgs in CASES:
        state: ShopSenseState = {"messages": [HumanMessage(content="q"), *msgs]}
        out = output_rail_node(state)
        flags = out.get("gate_flags") or ["(clean)"]
        print(f"  {label}")
        print(f"     {'; '.join(flags)}")
        if out.get("messages"):
            print(f"     rewritten -> {out['messages'][0].text[:70]}")
