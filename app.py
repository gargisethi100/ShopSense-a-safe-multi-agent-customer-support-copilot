"""The Streamlit UI: chat, the refund approval panel, and the cost footer.

THE ONE THING TO UNDERSTAND ABOUT STREAMLIT
    On EVERY interaction - a message sent, a button clicked, a widget
    touched - Streamlit re-runs this entire file from line 1. There is no
    event handler, no callback tree, no partial update: the script runs
    again and whatever it draws is the new page.

    That sounds wasteful and is actually why this file is short. The page
    is a pure function of state, so there is no UI state to keep in sync
    with the data. Two consequences shape everything below:

      1. Anything expensive must be cached (@st.cache_resource), or you
         would open a new database pool on every keystroke.
      2. Anything that must survive a rerun goes in st.session_state.

WHY THE INTERRUPT MAPS ONTO THIS PERFECTLY
    A LangGraph interrupt is "stop, return, wait for a decision". A
    Streamlit rerun is "the page just re-drew because a human did
    something". They are the same shape. So the refund gate needs no
    polling and no websockets: the run pauses, the script ends, the
    buttons render, and a click starts a new run that resumes it.

WHERE THE MESSAGES COME FROM
    Not from a list in session_state - from the CHECKPOINTER. The graph's
    saved state is the single source of truth, so a refresh, a reconnect,
    or a second browser tab all show the same conversation. Keeping a
    parallel copy in the UI would be a second source of truth, and the two
    would eventually disagree.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from config import get_settings
from graph.build import build_graph, get_checkpointer
from graph.memory import save_profile
from graph.state import format_cost_footer
from obs.costlog import record_run, trace_config, tracing_status

st.set_page_config(page_title="ShopSense Support", page_icon="🛒", layout="centered")


@st.cache_resource
def get_graph():
    """Built ONCE per server process, not once per rerun.

    Without the cache this would open a fresh connection pool on every
    keystroke - the fastest way to exhaust a database's connection limit
    and the single most common Streamlit performance bug.
    """
    return build_graph(checkpointer=get_checkpointer())


def new_thread() -> None:
    """Start a fresh conversation, saving the old one's profile first."""
    if (old := st.session_state.get("thread_id")) and st.session_state.get("logged"):
        try:
            state = get_graph().get_state(trace_config(old)).values
            if state.get("customer_id"):
                save_profile(state)
        except Exception:
            # A failed profile save must never block starting a new chat.
            pass
    st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
    st.session_state.logged = 0
    st.session_state.pending = None


if "thread_id" not in st.session_state:
    new_thread()

graph = get_graph()
config = trace_config(st.session_state.thread_id)
settings = get_settings()


# ---------------------------------------------------------------------------
# Sidebar - the operator's view
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Session")
    st.code(st.session_state.thread_id, language=None)
    st.button("New conversation", on_click=new_thread, use_container_width=True)

    state = graph.get_state(config).values
    if cid := state.get("customer_id"):
        st.caption(f"identified as **{cid}**")
    if profile := state.get("profile_summary"):
        with st.expander("Remembered from past visits"):
            st.write(profile)
    if summary := state.get("summary"):
        with st.expander("Earlier in this conversation (summarised)"):
            st.write(summary)

    st.divider()
    st.subheader("Safety")
    mode = settings.guardrails_mode
    st.caption(f"guardrails: **{mode}**"
               + ("" if mode == "enforce" else " — logging, not blocking"))
    # Showing the guardrail flags is deliberate: a safety layer nobody can
    # see is a safety layer nobody maintains.
    if flags := state.get("gate_flags"):
        with st.expander(f"Guardrail activity ({len(flags)})"):
            for f in flags:
                st.caption(f"• {f}")

    st.divider()
    st.caption(f"tracing: {tracing_status()}")


# ---------------------------------------------------------------------------
# Transcript - rendered from the checkpointer, never from a UI-side list
# ---------------------------------------------------------------------------

st.title("🛒 ShopSense Support")

for m in state.get("messages", []):
    if isinstance(m, HumanMessage):
        st.chat_message("user").write(m.text)
    elif isinstance(m, AIMessage) and not m.tool_calls:
        # Tool calls and tool results are machinery, not conversation.
        # They belong in the trace, not in front of a customer.
        st.chat_message("assistant").write(m.text)

if state.get("usage"):
    st.caption(f"_{format_cost_footer(state)}_")


# ---------------------------------------------------------------------------
# The refund approval panel - what the whole interrupt machinery was for
# ---------------------------------------------------------------------------


def resume(approved: bool) -> None:
    """Button handler: unfreeze the graph with a human's decision."""
    st.session_state.decision = {
        "approved": approved,
        # In a real deployment this is the signed-in operator, taken from
        # the session - never typed in, and never defaulted. The refunds
        # table requires a name because SOMEONE must be accountable.
        "approved_by": "web-operator",
        "note": "" if approved else "declined in approval panel",
    }


if pending := st.session_state.get("pending"):
    r = pending["refund"]
    with st.container(border=True):
        st.subheader("⏸️ Human approval required")
        st.caption("The agent has paused. Nothing has been written yet.")
        c1, c2 = st.columns(2)
        c1.metric("Refund amount", f"${r['amount_usd']}")
        c2.metric("Order", r["order_id"])
        st.write(f"**Customer:** {r['customer_name']} ({r['customer_id']})")
        st.write(f"**Item:** {r['product_name']}")
        st.write(f"**Reason given:** {r['reason']}")
        st.code(r["refund_id"], language=None)

        b1, b2 = st.columns(2)
        b1.button("Approve refund", type="primary", use_container_width=True,
                  on_click=resume, args=(True,))
        b2.button("Reject", use_container_width=True, on_click=resume, args=(False,))


# ---------------------------------------------------------------------------
# Driving the graph
# ---------------------------------------------------------------------------


def run(payload) -> None:
    """Invoke or resume, then park any new interrupt for the next rerun."""
    with st.spinner("Working..."):
        result = graph.invoke(payload, config)

    if "__interrupt__" in result:
        # The run froze. Store the payload and let the script end - the
        # rerun draws the approval panel. No polling, no background thread:
        # the pause and the page redraw are the same event.
        st.session_state.pending = result["__interrupt__"][0].value
    else:
        st.session_state.pending = None
        record_run(
            result,
            thread_id=st.session_state.thread_id,
            since=st.session_state.logged,
        )
        st.session_state.logged = len(result.get("usage") or [])


# A decision was recorded by a button click on the PREVIOUS rerun.
if decision := st.session_state.pop("decision", None):
    run(Command(resume=decision))
    st.rerun()

# Chat input is disabled while an approval is outstanding: letting the
# customer keep typing into a frozen graph would queue messages that
# cannot be processed, and imply the conversation is still moving.
if prompt := st.chat_input(
    "Ask about an order, or our return policy...",
    disabled=bool(st.session_state.get("pending")),
):
    run({"messages": [HumanMessage(content=prompt)]})
    st.rerun()

if st.session_state.get("pending"):
    st.info("Waiting on a human decision above. The conversation resumes "
            "the moment someone clicks.")
