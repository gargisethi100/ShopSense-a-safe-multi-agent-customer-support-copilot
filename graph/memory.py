"""Memory: keep conversations short, and remember customers between them.

TWO DIFFERENT PROBLEMS, TWO DIFFERENT MECHANISMS

  1. WITHIN a conversation - context bloat.
     Every turn resends the whole transcript, so a long chat costs more and
     more per message and eventually overflows the context window. Fix:
     once the transcript passes a threshold, fold the OLD messages into a
     short summary and delete them. The summary rides along instead.

  2. BETWEEN conversations - amnesia.
     Checkpoints are per thread_id; a customer returning tomorrow with a
     new thread starts from nothing. Fix: after a session, distil the
     durable facts about that customer into user_profiles, and load them
     at the start of their next visit.

MEMORY IS CURATION, NOT LOGGING
    Both mechanisms THROW INFORMATION AWAY on purpose. A memory that keeps
    everything is just the transcript again, with worse retrieval. The
    engineering is in deciding what survives - which is why summarisation
    here carries an explicit preserve-list (order ids, amounts, dates,
    promises made), and why the profile stores traits rather than events.

WHY THE CHEAP MODEL
    Both calls are compression, not reasoning: read some text, keep the
    load-bearing parts. That is exactly the router model's job description,
    and doing it on the expensive model would tax every long conversation.

Run directly to exercise both paths (needs Bedrock + the database):

    python -m graph.memory
"""

from __future__ import annotations

import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)

from config import get_settings
from graph.state import ShopSenseState
from llm import get_llm, usage_from

# Summarise once the transcript exceeds this many messages. Counting
# MESSAGES (not tokens) is deliberately crude: it is stable, testable, and
# free. Tokens would be more precise and would need a call to measure.
SUMMARY_THRESHOLD = 12

# How many recent messages stay verbatim. Recent turns carry the pronouns
# ("it", "that one") that a summary flattens away, so the tail must survive
# intact or the next answer loses the thread.
KEEP_RECENT = 6

SUMMARY_PROMPT = """You are compressing an ongoing customer-support \
conversation so it can continue without the full transcript.

Write a short summary (under 150 words) of what has happened so far.

PRESERVE EXACTLY, never paraphrase or round:
- order ids, customer ids, refund ids, tracking numbers
- amounts of money, dates, and deadlines
- anything the assistant PROMISED or told the customer would happen
- unresolved questions the customer is still waiting on

Drop: pleasantries, restated questions, and any detail that would not \
change what happens next. Write plain prose in the third person \
("The customer asked about..."), not a transcript."""

PROFILE_PROMPT = """You maintain a long-lived profile of a customer for a \
support team. You will be given the existing profile (possibly empty) and a \
conversation that just ended.

Return the UPDATED profile as JSON with exactly these keys:
  "summary": one or two sentences on who this customer is and what matters \
to them across visits
  "facts": a list of short durable statements (max 6)

Include only things that will STILL BE TRUE and USEFUL next month: stated \
preferences, recurring problems, products they own, communication style, \
outcomes of past disputes.

Exclude: one-off order statuses, anything already resolved, and anything \
the customer did not actually say. If the conversation adds nothing \
durable, return the existing profile unchanged.

Never invent facts. Return JSON only - no prose, no code fences.

BEWARE COMPOUNDING. The existing profile already accounts for every past \
conversation, so the new conversation is not additional evidence for what \
it already says. If both mention the same thing, that is ONE fact, not two: \
never increase a count, escalate a severity, or add an occurrence unless \
this conversation states a NEW one explicitly. When in doubt, keep the \
existing wording."""


# ---------------------------------------------------------------------------
# 1. Rolling summarisation (within a conversation)
# ---------------------------------------------------------------------------


def _safe_cut(messages: list, keep_recent: int) -> int:
    """Where can we cut the transcript WITHOUT breaking it?

    A ToolMessage is only valid immediately after the AIMessage whose tool
    call it answers. Cut carelessly and the surviving history can begin
    with an orphan tool result - which the API rejects outright, turning a
    cost optimisation into an outage.

    So we walk backwards from the ideal cut point to the nearest
    HumanMessage, which is always a safe boundary: a customer turn never
    depends on what came before it structurally.
    """
    ideal = len(messages) - keep_recent
    for i in range(ideal, 0, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return 0  # no safe boundary found -> summarise nothing


def summarize_node(state: ShopSenseState) -> dict:
    """Fold old messages into state['summary'] once the transcript is long.

    Runs at the FRONT of every turn, so the compression happens before the
    supervisor and specialists pay to read the transcript - not after.
    """
    messages = state.get("messages") or []
    if len(messages) <= SUMMARY_THRESHOLD:
        return {}

    cut = _safe_cut(messages, KEEP_RECENT)
    if cut <= 0:
        return {}

    old, previous = messages[:cut], state.get("summary")
    settings = get_settings()
    llm = get_llm("router")

    prior = f"\n\nSummary of even earlier turns:\n{previous}" if previous else ""
    reply = llm.invoke(
        [
            SystemMessage(content=SUMMARY_PROMPT + prior),
            *old,
            HumanMessage(content="Summarise the conversation above."),
        ]
    )
    u = usage_from(reply, settings.model_router)

    return {
        "summary": reply.text,
        # RemoveMessage is how you DELETE from a reducer-managed list: the
        # add_messages reducer sees these and drops the matching ids.
        # Returning new content and deletions in one update is fine - the
        # reducer applies both.
        "messages": [RemoveMessage(id=m.id) for m in old],
        "usage": [
            {
                "node": "summarize",
                "model": settings.model_router,
                "input_tokens": u.total_input,
                "output_tokens": u.output,
                "cost_usd": u.cost,
            }
        ],
        "gate_flags": [f"summarised {len(old)} messages into state['summary']"],
    }


def summary_preamble(state: ShopSenseState) -> list:
    """The summary + profile, as a message the specialists can read.

    Kept OUT of state['messages'] deliberately: it is context, not
    conversation. Nodes splice it in at call time so it never gets
    summarised into itself, and never shows up in the customer transcript.
    """
    parts = []
    if profile := state.get("profile_summary"):
        parts.append(f"What we know about this customer from past visits:\n{profile}")
    if summary := state.get("summary"):
        parts.append(f"Earlier in this conversation:\n{summary}")
    return [SystemMessage(content="\n\n".join(parts))] if parts else []


# ---------------------------------------------------------------------------
# 2. Per-customer profiles (between conversations)
# ---------------------------------------------------------------------------


def load_profile(customer_id: str) -> str | None:
    """Read a customer's profile as a plain paragraph, or None."""
    from db.pool import get_graph_pool

    with get_graph_pool().connection() as conn:
        row = conn.execute(
            "SELECT profile FROM user_profiles WHERE customer_id = %s",
            (customer_id,),
        ).fetchone()

    if not row or not row["profile"]:
        return None
    p = row["profile"]
    lines = [p.get("summary", "")]
    lines += [f"- {f}" for f in p.get("facts", [])]
    return "\n".join(x for x in lines if x) or None


def save_profile(state: ShopSenseState) -> str | None:
    """Distil the finished conversation into the customer's profile.

    Called AFTER a session, not during one - so its latency and cost never
    sit between a customer and their answer. Returns the new profile text,
    or None if there was nothing to save.
    """
    customer_id = state.get("customer_id")
    messages = state.get("messages") or []
    if not customer_id or len(messages) < 2:
        return None

    from db.pool import get_graph_pool

    existing = load_profile(customer_id) or "(no profile yet)"

    # Send only the CONVERSATION, not the machinery. Tool calls and tool
    # results are how the answer was found; they say nothing durable about
    # the customer, they cost tokens, and passing them to a model with no
    # tools bound makes the provider rewrite them (a warning, and a subtly
    # different prompt than we intended). Filtering is cheaper and clearer.
    dialogue = [
        m
        for m in messages
        if isinstance(m, HumanMessage)
        or (isinstance(m, AIMessage) and not m.tool_calls)
    ]
    if not dialogue:
        return None

    reply = get_llm("router").invoke(
        [
            SystemMessage(content=PROFILE_PROMPT),
            HumanMessage(content=f"EXISTING PROFILE:\n{existing}"),
            *dialogue,
            HumanMessage(content="Return the updated profile JSON."),
        ]
    )

    try:
        # Models sometimes wrap JSON in prose or fences despite instructions;
        # take the outermost braces rather than trusting the whole string.
        raw = reply.text
        profile = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
        if not isinstance(profile.get("facts", []), list):
            raise ValueError("facts must be a list")
    except (ValueError, json.JSONDecodeError):
        # A malformed profile is not worth a crash OR a corrupted store.
        # Keeping the old profile is always a safe outcome.
        return None

    with get_graph_pool().connection() as conn:
        # UPSERT: insert, or overwrite if this customer already has a row.
        # The primary key on customer_id (Phase 1) is what makes this one
        # statement instead of a read-then-write race.
        conn.execute(
            "INSERT INTO user_profiles (customer_id, profile, updated_at)"
            " VALUES (%s, %s, now())"
            " ON CONFLICT (customer_id) DO UPDATE"
            " SET profile = EXCLUDED.profile, updated_at = now()",
            (customer_id, json.dumps(profile)),
        )
    return load_profile(customer_id)


def hydrate_profile(state: ShopSenseState) -> dict:
    """Load the profile once the customer becomes known mid-conversation."""
    cid = state.get("customer_id")
    if not cid or state.get("profile_summary"):
        return {}
    profile = load_profile(cid)
    return {"profile_summary": profile} if profile else {}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import enable_utf8_console

    enable_utf8_console()

    # A transcript long enough to trigger compression.
    #
    # NOTE THE ORDERING, which the first version of this test got wrong:
    # the load-bearing facts must sit EARLY, in the part that actually gets
    # summarised. Put them in the last few messages and they survive
    # because they were never compressed - a test that passes for the
    # wrong reason, which is worse than one that fails.
    convo: list = [
        HumanMessage(content="what about order ord_1003?"),
        AIMessage(content="ord_1003 was delivered Jul 23; the $79.99 refund is approved."),
        HumanMessage(content="great, when do I see the money?"),
        AIMessage(content="I promised it lands within 5-10 business days."),
    ]
    for i in range(5):
        convo += [
            HumanMessage(content=f"unrelated small talk {i}"),
            AIMessage(content=f"friendly reply {i}"),
        ]

    print(f"=== summarisation ({len(convo)} messages, threshold {SUMMARY_THRESHOLD})")
    out = summarize_node({"messages": convo})
    if not out:
        print("  under threshold - nothing to do")
    else:
        removed = len(out["messages"])
        print(f"  removed {removed} old messages, kept {len(convo) - removed} verbatim")
        print(f"  cost   : ${out['usage'][0]['cost_usd']:.5f}")
        print(f"  summary: {out['summary']}")
        # The real question is not "is it in the summary" but "does it
        # survive ANYWHERE the model will still see" - summary OR the
        # verbatim tail. That is the property the system depends on.
        removed_ids = {m.id for m in out["messages"]}
        survivors = out["summary"] + "\n".join(
            m.text for m in convo if m.id not in removed_ids
        )
        print("\n  preserve-list check (summary + kept messages):")
        for token in ("ord_1003", "79.99", "5-10"):
            hit = token in survivors
            print(f"    {token:10} {'kept' if hit else 'LOST  <- prompt needs work'}")

    print("\n=== profile round-trip (customer cust_001)")
    state: ShopSenseState = {
        "customer_id": "cust_001",
        "messages": [
            HumanMessage(content="Hi, it's Dana. Please always email me, I never "
                                 "answer the phone. My Aurora headphones broke "
                                 "again - second pair this year."),
            AIMessage(content="Noted - I'll keep contact by email only."),
        ],
    }
    saved = save_profile(state)
    print(f"  saved  : {saved}")
    print(f"  reload : {load_profile('cust_001')}")
