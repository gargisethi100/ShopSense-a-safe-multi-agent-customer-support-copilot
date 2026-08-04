"""The supervisor node: reads the conversation, picks who acts next.

THE PATTERN: "LLM WITH A MENU"
    The supervisor is not a chatbot. It is a CLASSIFIER with three legal
    answers - order_agent, policy_agent, FINISH - and its output feeds a
    conditional edge, i.e. an if-statement whose condition is a model's
    choice. That is the whole idea of multi-agent routing.

WHY A SEPARATE ROUTER AT ALL (instead of one big agent with every tool)
    * Focus. Each specialist sees only its own tools and its own
      instructions, so it can't pick a policy tool for an order question.
      Fewer choices -> fewer wrong choices.
    * Cost. Routing is a high-volume, low-difficulty call, so it runs on
      the cheap model (haiku) while the specialists run on the smart one.
    * Testability. "question -> expected specialist" is a table you can
      assert on. That table IS the Phase 10 routing eval, and it is why a
      prompt edit can be caught before it ships.

THE THREE RELIABILITY DECISIONS (each maps to code below)
    1. STRUCTURED OUTPUT, not free text. We bind a schema, so the answer
       arrives as a validated object with a Route field - never a chatty
       "I think the order agent should handle this!" that we'd have to
       parse with a regex.
    2. TEMPERATURE 0 where supported. Same question -> same route. This is
       what makes routing evals stable instead of flaky.
    3. A LOOP GUARD. The graph will come BACK here after each specialist
       (that is how multi-step answers work). A model that keeps saying
       "order_agent" would spin forever and bill forever. After a hard cap
       we stop asking and force FINISH.

Run directly to route real questions against Bedrock (costs a few cents):

    python -m graph.supervisor
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import get_settings
from graph.state import Route, ShopSenseState
from llm import get_llm, usage_from

# How many specialist hops before we stop asking and force an answer.
# 6 covers the realistic worst case (identify customer -> list orders ->
# order details -> policy check -> refund) with room to spare, while making
# an infinite loop impossible. A cap you never hit still has to exist.
MAX_HOPS = 6


class RoutingDecision(BaseModel):
    """The supervisor's only legal output shape.

    The field DESCRIPTIONS are sent to the model as part of the schema, so
    this class is simultaneously: the prompt that explains the choices, the
    validator that rejects anything else, and the type the rest of the
    graph reads. One definition, three jobs.
    """

    reasoning: str = Field(
        description=(
            "One short sentence explaining the choice. Write this FIRST - "
            "deciding out loud before committing improves the choice, and "
            "it lands in the trace so a wrong route is debuggable."
        )
    )
    next: Route = Field(
        description=(
            "Who acts next. "
            "'order_agent' for anything about a specific customer, order, "
            "delivery status, tracking, or a refund request. "
            "'policy_agent' for questions about the rules: returns, "
            "warranties, shipping costs and times, what is allowed. "
            "'FINISH' when the last assistant message already answers the "
            "customer, or the request needs neither specialist."
        )
    )


SYSTEM_PROMPT = """You are the supervisor of a customer-support team for an \
online store. You do not talk to the customer. You read the conversation and \
decide which specialist acts next.

Your team:
- order_agent: has database access to customers, orders, and delivery status, \
and can start a refund (which a human then approves). Use it for anything \
about a SPECIFIC order or customer.
- policy_agent: has the published policy documents (returns, shipping, \
warranty) and answers what the RULES are, with citations.

Choose FINISH when the last assistant message already answers the customer's \
question, when the customer is only greeting or thanking you, or when the \
request is outside what this team can do.

Guidance:
- Route on what is still MISSING, not on what has already been done. If a \
specialist has just supplied the missing piece and nothing else is needed, \
FINISH.
- A question can need both specialists in turn: e.g. "can I return the \
headphones I got last week?" needs the policy rule AND the order's delivery \
date. Route to one now; you will be asked again after it reports back.
- Never route to a specialist to repeat work that is already in the \
conversation."""


def supervisor_node(state: ShopSenseState) -> dict:
    """One routing decision. Returns only the fields it changed.

    Read this as the template every node follows:
        1. look at state
        2. (maybe) call an LLM
        3. return a small dict of CHANGES - never the whole state
    """
    settings = get_settings()
    messages = state.get("messages") or []

    # --- the loop guard, before spending a token -------------------------
    # Count how many times we've already routed in this conversation turn.
    hops = len([r for r in (state.get("usage") or []) if r["node"] == "supervisor"])
    if hops >= MAX_HOPS:
        return {
            "route": "FINISH",
            "gate_flags": [f"supervisor: hop limit {MAX_HOPS} reached, forced FINISH"],
        }

    # --- the call --------------------------------------------------------
    # temperature=0 only where the model allows it. The capability table in
    # config.py decides; this node cannot get it wrong. (Haiku accepts it,
    # which is exactly why the router lives there.)
    kwargs = {}
    if settings.supports_temperature(settings.model_router):
        kwargs["temperature"] = 0.0
    llm = get_llm("router", **kwargs)

    # with_structured_output binds the schema: the model must answer with a
    # RoutingDecision, and LangChain validates it before we ever see it.
    router = llm.with_structured_output(RoutingDecision, include_raw=True)

    result = router.invoke([SystemMessage(content=SYSTEM_PROMPT), *messages])
    decision: RoutingDecision = result["parsed"]

    # include_raw=True gives us the underlying AIMessage too - which is the
    # only place token usage lives. Without it, routing would silently cost
    # money that never appears in the footer.
    usage = usage_from(result["raw"], settings.model_router)

    return {
        "route": decision.next,
        "usage": [
            {
                "node": "supervisor",
                "model": settings.model_router,
                "input_tokens": usage.total_input,
                "output_tokens": usage.output,
                "cost_usd": usage.cost,
            }
        ],
        # NOTE: we deliberately do NOT append the routing decision to
        # `messages`. The customer's transcript should contain the
        # conversation, not our internal dispatch notes - those belong in
        # the trace (LangSmith, Phase 8), where debugging happens.
    }


def route_from_state(state: ShopSenseState) -> str:
    """The conditional edge's condition function.

    LangGraph calls this AFTER supervisor_node and uses the returned string
    to pick the next node. It is deliberately dumb - all the thinking
    already happened; this just reads the signpost. Defaulting to FINISH
    means a missing/garbled route ends the turn instead of hanging.
    """
    return state.get("route") or "FINISH"


# ---------------------------------------------------------------------------
# Smoke test: the routing table, run for real. This is the Phase 10 eval in
# embryo - same shape, just printed instead of asserted.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    CASES: list[tuple[str, Route, list]] = [
        # (question, expected route, prior conversation)
        ("Where is my order ord_1003?", "order_agent", []),
        ("How long do I have to return something?", "policy_agent", []),
        ("Is shipping free?", "policy_agent", []),
        ("My headphones arrived cracked, I want my money back.", "order_agent", []),
        ("hi there!", "FINISH", []),
        ("what's the capital of France?", "FINISH", []),
        # A conversation where the work is already done -> must FINISH,
        # not route again. This is the case that catches loop-y prompts.
        (
            "thanks!",
            "FINISH",
            [
                HumanMessage(content="where is ord_1003?"),
                AIMessage(content="Order ord_1003 was delivered on Jul 23."),
            ],
        ),
    ]

    print(f"router model: {get_settings().model_router}\n")
    passed = 0
    total_cost = 0.0
    for question, expected, prior in CASES:
        state: ShopSenseState = {
            "messages": [*prior, HumanMessage(content=question)]
        }
        out = supervisor_node(state)
        got = out["route"]
        cost = out["usage"][0]["cost_usd"]
        total_cost += cost
        ok = got == expected
        passed += ok
        print(f"  [{'ok  ' if ok else 'MISS'}] {question[:44]:46} -> {got:12} "
              f"(expected {expected}, ${cost:.5f})")

    print(f"\n{passed}/{len(CASES)} routed as expected. total cost ${total_cost:.4f}")
    print("Misses are not bugs yet - they are prompt feedback. Phase 10 turns")
    print("this table into a pytest that BLOCKS the deploy when it regresses.")
