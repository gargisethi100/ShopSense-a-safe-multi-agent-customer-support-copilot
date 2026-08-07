"""The policy specialist: answers the RULES, and cites where each rule lives.

SAME SHAPE, DIFFERENT JOB
    This node is deliberately the twin of order_agent.py: same ReAct loop,
    same usage accounting, same bounded rounds. Read them side by side -
    what differs is only the TOOLS it holds and the STANDARD it is held to.
    Once the loop is a template, adding a specialist is a prompt and a
    toolbelt, not new machinery.

THE STANDARD: GROUNDEDNESS
    An order answer is checkable against a database row. A policy answer is
    checkable against a document - but only if it says WHICH document. So
    this specialist has one non-negotiable rule: every policy claim carries
    its section id, e.g. "[RET-1]". That turns an assertion into a pointer
    a human (or the Phase 7 output rail) can verify.

    Why this matters more than it sounds: a wrong order status is an
    inconvenience. A confidently invented return policy is the store making
    a promise it never made - a contractual problem, not a support one.

THREE DEFENCES AGAINST INVENTED POLICY, STACKED
    1. The tool result opens with the citation contract (tools/policy_tools)
    2. The system prompt below repeats it as an identity, not a request
    3. Phase 7's output rail will flag uncited policy claims from outside
    None of the three is sufficient alone; models are good at slipping past
    exactly one instruction.

Run directly (needs Bedrock; no database):

    python -m agents.policy_agent
"""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from config import get_settings
from graph.state import ShopSenseState
from llm import get_llm, usage_from
from tools.policy_tools import POLICY_TOOLS

TOOLS = list(POLICY_TOOLS)
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# Lower than the order agent's 3. A policy question is usually one search;
# two allows a legitimate second topic ("can I return it AND is postage
# free?"). Beyond that the model is fishing, not researching.
MAX_TOOL_ROUNDS = 2

SYSTEM_PROMPT = """You are the policy specialist for an online store's support \
team. You answer questions about the store's published rules: returns, \
refunds, shipping, and warranty.

You have exactly one tool: search_policies. Use it before answering ANY \
policy question. You have no memory of these policies - only the excerpts \
the tool returns are real.

Non-negotiable rules:
- Cite the section id next to every policy fact you state, like this: \
"You have 30 days from delivery to return an item [RET-1]."
- If the excerpts do not cover what was asked, say the published policies \
do not cover it and offer to connect the customer with human support. Never \
fill a gap with what is "usually" true elsewhere.
- Never soften or round a policy to be friendlier. "30 days" does not become \
"about a month"; a 14-day damage-report deadline does not disappear because \
the return window is longer.
- If two rules interact (e.g. a general window and a shorter deadline for a \
special case), state BOTH and cite both.

How to answer:
- Answer the question first, in one or two sentences, then the conditions \
that matter. Warm and brief; the customer wants the rule, not an essay.
- If the answer depends on facts about THEIR order (when it arrived, what it \
cost, whether it shipped), say which specific detail is needed - the order \
specialist can look it up. Do not guess their order details."""


def policy_agent_node(state: ShopSenseState) -> dict:
    """Run the ReAct loop until Claude writes a cited answer (or we cap it)."""
    settings = get_settings()
    llm = get_llm("agent").bind_tools(TOOLS)

    convo = [SystemMessage(content=SYSTEM_PROMPT), *(state.get("messages") or [])]
    new_messages: list = []
    usage_records: list[dict] = []
    flags: list[str] = []

    for round_no in range(MAX_TOOL_ROUNDS + 1):
        reply: AIMessage = llm.invoke(convo)

        u = usage_from(reply, settings.model_agent)
        usage_records.append(
            {
                "node": "policy_agent",
                "model": settings.model_agent,
                "input_tokens": u.total_input,
                "output_tokens": u.output,
                "cost_usd": u.cost,
            }
        )
        convo.append(reply)
        new_messages.append(reply)

        if not reply.tool_calls:
            break

        if round_no == MAX_TOOL_ROUNDS:
            flags.append(f"policy_agent: tool round cap ({MAX_TOOL_ROUNDS}) hit")
            stop = ToolMessage(
                content=(
                    "SEARCH BUDGET EXHAUSTED for this turn. Answer from the "
                    "excerpts you already have, citing them, and say plainly "
                    "which part of the question you could not verify."
                ),
                tool_call_id=reply.tool_calls[0]["id"],
            )
            convo.append(stop)
            new_messages.append(stop)
            continue

        for call in reply.tool_calls:
            tool = TOOLS_BY_NAME.get(call["name"])
            if tool is None:
                # The likeliest hallucination here is an ORDER tool - this
                # specialist can see order questions in the transcript but
                # holds no order tools. Name the boundary explicitly so it
                # hands back rather than improvising.
                result = (
                    f"UNKNOWN TOOL '{call['name']}'. You only have: "
                    f"{', '.join(TOOLS_BY_NAME)}. If the question needs order "
                    "details, say which detail is needed and stop - the order "
                    "specialist will handle it."
                )
                flags.append(f"policy_agent: hallucinated tool '{call['name']}'")
            else:
                try:
                    result = tool.invoke(call["args"])
                except RuntimeError as e:
                    result = (
                        f"SYSTEM NOT CONFIGURED: {e} Do not retry. Apologise "
                        "and offer human support."
                    )
                    flags.append(f"policy_agent: config error on {call['name']}")
                except Exception as e:
                    result = (
                        f"TOOL ARGUMENT ERROR: {e}. Re-read the tool's "
                        "description and try again with corrected arguments."
                    )
                    flags.append(f"policy_agent: bad args for {call['name']}")

            msg = ToolMessage(content=str(result), tool_call_id=call["id"])
            convo.append(msg)
            new_messages.append(msg)

    return {
        "messages": new_messages,
        "usage": usage_records,
        **({"gate_flags": flags} if flags else {}),
    }


# ---------------------------------------------------------------------------
# Smoke test. Note the last two cases: they are the ones that catch a model
# being helpful instead of accurate.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import re

    from config import enable_utf8_console

    enable_utf8_console()

    QUESTIONS = [
        "How long do I have to return something?",
        "My headphones died after 3 months - are they covered?",
        "Is delivery free?",
        # The trap: the 30-day return window is NOT the answer here. Damage
        # claims have their own 14-day deadline [RET-3]. A model that skims
        # answers "30 days" and is wrong in the customer's favour.
        "My order arrived smashed. How long do I have to report it?",
        # Not in the corpus at all -> must decline, not improvise.
        "Do you price match with other stores?",
    ]

    CITATION = re.compile(r"\[[A-Z]+-\d+\]")
    # Excerpt headers look like: '--- [RET-1] Return window (returns.md) ---'.
    # Match THOSE, not every bracketed id in the result - the citation
    # contract line contains a "[RET-1]" as an EXAMPLE, and counting it as a
    # retrieval hit made the first run look like RET-1 was found when it
    # wasn't. A debugging display that lies costs more than no display.
    RETRIEVED = re.compile(r"^--- \[([A-Z]+-\d+)\]", re.MULTILINE)

    print(f"agent model: {get_settings().model_agent}\n")
    total = 0.0
    for q in QUESTIONS:
        print(f"=== {q}")
        state: ShopSenseState = {"messages": [HumanMessage(content=q)]}
        out = policy_agent_node(state)

        for m in out["messages"]:
            if isinstance(m, AIMessage) and m.tool_calls:
                for c in m.tool_calls:
                    print(f"  tool  : {c['name']}({c['args']})")
            elif isinstance(m, ToolMessage):
                sections = RETRIEVED.findall(str(m.content))
                print(f"  found : {', '.join(sections) or '(nothing relevant)'}")

        answer = out["messages"][-1].text
        cited = CITATION.findall(answer)
        print(f"  ANSWER: {answer}")
        # A crude groundedness check - exactly the shape the Phase 7 output
        # rail and the Phase 10 evals will formalise.
        print(f"  cites : {', '.join(cited) if cited else 'NONE (ok only if declining)'}")

        cost = sum(r["cost_usd"] for r in out["usage"])
        total += cost
        print(f"  ({len(out['usage'])} LLM calls, ${cost:.4f})\n")

    print(f"total spend this run: ${total:.4f}")
