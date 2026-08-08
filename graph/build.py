"""Wire the nodes into a runnable graph, and give it a memory.

THE FLOWCHART WE HAVE BEEN BUILDING TOWARD

        START
          |
          v
    +-----------+      route == "order_agent"     +-------------+
    | supervisor|-------------------------------->| order_agent |--+
    |           |      route == "policy_agent"    +-------------+  |
    |           |----------------------------+                     |
    +-----------+                            v                     |
          |  route == "FINISH"        +--------------+             |
          v                           | policy_agent |--+          |
         END                          +--------------+  |          |
                                                        |          |
          ^-------------- back to supervisor -----------+----------+

    The loop back is what makes multi-step answers possible: a specialist
    reports, the supervisor looks again, and either dispatches the other
    specialist or ends the turn. The hop cap in supervisor.py is what stops
    that loop being infinite.

TWO KINDS OF EDGE
    * add_edge(a, b)              always go a -> b
    * add_conditional_edges(a, f) call f(state) and use its ANSWER to pick
                                  the next node. This is the if-statement
                                  whose condition is a model's decision.

THE CHECKPOINTER, AND THE DATABASE DECISION IT FORCED
    Compiling with a checkpointer makes the graph stateful: after every
    node, the entire state is written to Postgres under a `thread_id`.
    Send a message with the same thread_id tomorrow, from a different
    process, after a redeploy - the conversation continues.

    But that means a component now stores WHOLE CONVERSATIONS, so its
    privileges matter. Neither agent_ro nor refund_writer can create or
    write LangGraph's tables, and using the admin keycard at runtime would
    hand a live web app the power to DROP anything. So there is a third
    narrow role, graph_writer, that owns the checkpoint tables and CANNOT
    SEE customers, orders, or refunds at all.

    Setup (admin, once)  : python -m graph.build setup
    Runtime (graph_writer): everything else

Run:
    python -m graph.build setup    create checkpoint tables + grant the role
    python -m graph.build          chat with the whole system from the CLI
"""

from __future__ import annotations

import sys
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from langgraph.types import Command

from agents.order_agent import order_agent_node
from agents.policy_agent import policy_agent_node
from graph.approval import refund_approval_node, route_after_order_agent
from graph.memory import hydrate_profile, save_profile, summarize_node
from graph.state import ShopSenseState, format_cost_footer
from graph.supervisor import route_from_state, supervisor_node
from guards.input_gate import input_gate_node, route_after_gate
from guards.output_rail import output_rail_node
from obs.costlog import record_run, trace_config, tracing_status


def memory_node(state: ShopSenseState) -> dict:
    """One node, two memory jobs - both cheap, both before any reasoning."""
    return {**summarize_node(state), **hydrate_profile(state)}


def build_graph(checkpointer=None):
    """Assemble the flowchart. Pure wiring - no I/O, no model calls.

    Kept separate from the checkpointer so tests can compile a graph with
    no database at all: build_graph() with checkpointer=None is a complete,
    runnable (if forgetful) system.
    """
    builder = StateGraph(ShopSenseState)

    builder.add_node("input_gate", input_gate_node)
    builder.add_node("memory", memory_node)
    builder.add_node("output_rail", output_rail_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("order_agent", order_agent_node)
    builder.add_node("policy_agent", policy_agent_node)
    builder.add_node("refund_approval", refund_approval_node)

    # Every turn starts with memory: compress the transcript if it has grown
    # too long, and load the customer's profile if we now know who they are.
    # Doing this FIRST means the supervisor and specialists read the
    # compressed transcript - compressing afterwards would save nothing.
    # The guardrail sandwich. Nothing reaches the agents unscreened, and
    # nothing reaches the customer unchecked.
    builder.add_edge(START, "input_gate")
    builder.add_conditional_edges(
        "input_gate",
        route_after_gate,
        # A blocked message skips EVERYTHING - no memory, no supervisor, no
        # specialists. The gate already wrote the refusal; there is nothing
        # left to decide, and no reason to pay a model to agree.
        {"memory": "memory", "END": END},
    )
    builder.add_edge("memory", "supervisor")

    # The one conditional edge in the system. route_from_state() reads
    # state["route"] (already decided by supervisor_node) and returns a
    # string; the mapping turns that string into the next node. The mapping
    # is EXPLICIT so an unexpected value fails loudly instead of silently
    # picking a default.
    builder.add_conditional_edges(
        "supervisor",
        route_from_state,
        {
            "order_agent": "order_agent",
            "policy_agent": "policy_agent",
            # FINISH no longer means "done" - it means "ready to be
            # checked". The last thing before END is always the rail.
            "FINISH": "output_rail",
        },
    )
    builder.add_edge("output_rail", END)

    # Specialists always report back rather than answering the customer
    # directly. This is what allows a question to need both of them, and
    # it gives the supervisor one last look before the turn ends.
    #
    # The order agent has one detour: if it parked a refund proposal in
    # state, the run must pass through the human gate BEFORE anyone
    # summarises anything. Note this edge is data-driven, not model-driven -
    # the supervisor gets no say in whether a human is consulted.
    builder.add_conditional_edges(
        "order_agent",
        route_after_order_agent,
        {"refund_approval": "refund_approval", "supervisor": "supervisor"},
    )
    builder.add_edge("refund_approval", "supervisor")
    builder.add_edge("policy_agent", "supervisor")

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Checkpointer
# ---------------------------------------------------------------------------


def get_checkpointer() -> PostgresSaver:
    """Runtime checkpointer, on the narrow graph_writer role."""
    from db.pool import get_graph_pool

    return PostgresSaver(get_graph_pool())


def setup_checkpointer() -> None:
    """One-time, as ADMIN: create LangGraph's tables and hand them over.

    Two steps, and the second is the one people forget:
      1. PostgresSaver.setup() creates whatever tables THIS VERSION of the
         library needs. We never hand-write them - that is the library's
         business, and hand-written copies break on upgrade.
      2. GRANT those tables to graph_writer. Step 1 runs as admin, so admin
         OWNS the new tables; without step 2 the runtime role would find
         them unreadable.

    Safe to re-run: setup() is idempotent, and the grants are recomputed
    from whatever tables now exist - which also means re-running this after
    a library upgrade is the correct response to "new checkpoint table".
    """
    from psycopg.rows import dict_row
    from psycopg import sql

    from config import get_settings
    from db.pool import admin_connection

    import psycopg

    settings = get_settings()

    # PostgresSaver needs its own connection settings, so we open a
    # dedicated admin connection rather than reusing admin_connection().
    with psycopg.connect(
        settings.require_db_url("admin"), autocommit=True, row_factory=dict_row
    ) as conn:
        PostgresSaver(conn).setup()
        print("  checkpoint tables created/verified")

    with admin_connection() as conn:
        # Ensure the role exists, with the password from .env - same rule
        # as always: passwords come from the environment, never from a
        # committed file.
        from urllib.parse import urlparse

        pw = urlparse(settings.require_db_url("graph")).password
        exists = conn.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = 'graph_writer'"
        ).fetchone()
        conn.execute(
            sql.SQL("{verb} ROLE graph_writer WITH LOGIN PASSWORD {pw}").format(
                verb=sql.SQL("ALTER" if exists else "CREATE"), pw=sql.Literal(pw)
            )
        )
        print(f"  graph_writer role {'updated' if exists else 'created'}")

        # Find what setup() actually made. LangGraph's table names are its
        # own business and have changed between versions - so we DISCOVER
        # them rather than hard-coding a list that will rot.
        tables = [
            r[0]  # admin_connection uses the default tuple rows, not dict_row
            for r in conn.execute(
                """SELECT table_name FROM information_schema.tables
                   WHERE table_schema = 'public'
                     AND (table_name LIKE 'checkpoint%' OR table_name = 'writes')"""
            ).fetchall()
        ]
        if not tables:
            raise SystemExit("no checkpoint tables found after setup() - aborting")

        conn.execute("GRANT USAGE ON SCHEMA public TO graph_writer")
        for t in tables:
            conn.execute(
                sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON {} TO graph_writer")
                .format(sql.Identifier(t))
            )
        print(f"  granted on {len(tables)} table(s): {', '.join(sorted(tables))}")
        print("\n  NOTE what graph_writer still CANNOT do: read customers,")
        print("  orders, or refunds. It stores conversations, nothing else.")


# ---------------------------------------------------------------------------
# CLI chat - the whole system, before any UI exists
# ---------------------------------------------------------------------------


def chat() -> None:
    from config import enable_utf8_console, get_settings

    enable_utf8_console()

    graph = build_graph(checkpointer=get_checkpointer())
    thread_id = f"cli-{uuid.uuid4().hex[:8]}"
    # thread_id IS the conversation's identity. Same id -> same history,
    # loaded from Postgres. Change it and you are a different customer.
    # trace_config adds tags/metadata so the run is findable in LangSmith.
    config = trace_config(thread_id)

    print(f"ShopSense CLI  (thread {thread_id})")
    print(f"tracing: {tracing_status()}")
    print("Type a question, or 'quit'. Try:")
    print("  where is order ord_1003?")
    print("  how long do I have to return it?     <- tests memory across turns")
    print("-" * 60)
    logged = 0  # usage records already written to the cost log

    while True:
        try:
            text = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in {"quit", "exit"}:
            break

        # .invoke() runs the whole flowchart: supervisor -> specialist(s) ->
        # supervisor -> END, checkpointing after every step.
        result = graph.invoke({"messages": [HumanMessage(content=text)]}, config)

        # ...unless a node called interrupt(), in which case .invoke()
        # returns EARLY with the payload under "__interrupt__" and the run
        # is frozen in Postgres. Here we approve inline; in Phase 9 the
        # same payload renders as a panel with Approve/Reject buttons, and
        # the process could restart in between without losing anything.
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            r = payload["refund"]
            print("\n" + "=" * 60)
            print("  HUMAN APPROVAL REQUIRED - the graph is paused")
            print("=" * 60)
            print(f"  refund id : {r['refund_id']}")
            print(f"  customer  : {r['customer_name']} ({r['customer_id']})")
            print(f"  order     : {r['order_id']}  ({r['product_name']})")
            print(f"  amount    : ${r['amount_usd']}")
            print(f"  reason    : {r['reason']}")
            answer = input("  approve? [y/N] > ").strip().lower()
            decision = {
                "approved": answer in {"y", "yes"},
                "approved_by": "cli-operator",
                "note": "" if answer in {"y", "yes"} else "declined at CLI",
            }
            # Command(resume=...) restarts the interrupted node, with the
            # value above coming back OUT of the interrupt() call.
            result = graph.invoke(Command(resume=decision), config)

        # Show the internal steps, then the answer. In the Streamlit UI
        # (Phase 9) the steps get hidden and only the answer is shown.
        for m in result["messages"]:
            if isinstance(m, AIMessage) and m.tool_calls:
                for c in m.tool_calls:
                    print(f"    . {c['name']}({c['args']})")
            elif isinstance(m, ToolMessage):
                print(f"    . -> {str(m.content).splitlines()[0][:66]}")

        answer = next(
            (m for m in reversed(result["messages"])
             if isinstance(m, AIMessage) and not m.tool_calls),
            None,
        )
        print(f"\nbot > {answer.text if answer else '(no answer produced)'}")
        print(f"      [{format_cost_footer(result)}]")

        # One line per turn, appended locally. The footer above is
        # CUMULATIVE (what this conversation has cost so far - what the
        # customer sees); the log records only THIS turn's delta, which is
        # why `logged` is threaded through. Same numbers, two questions.
        record_run(result, thread_id=thread_id, question=text, since=logged)
        logged = len(result.get("usage") or [])

    # SESSION END. Distilling the profile here - after the customer has
    # gone - is the whole point: it costs a model call, and no customer
    # should ever wait for it. The conversation itself is already safe in
    # Postgres; this is the part that outlives the thread.
    final = graph.get_state(config).values
    if final.get("customer_id"):
        print("\nupdating customer profile...", end=" ", flush=True)
        profile = save_profile(final)
        print("done" if profile else "nothing durable to save")
        if profile:
            print("\n  " + profile.replace("\n", "\n  "))

    print("\nConversation saved. Resume it any time with the same thread_id.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        print("setting up the checkpointer (as admin)...")
        setup_checkpointer()
    else:
        chat()
