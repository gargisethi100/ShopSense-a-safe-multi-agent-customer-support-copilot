# ShopSense

A multi-agent customer-support system for an e-commerce store. A customer
asks about an order or a policy; a supervisor decides who should answer;
specialists look things up in Postgres or in the policy corpus; and any
refund stops at a human before money moves.

Built with LangGraph, FastAPI, Postgres, and Claude models on AWS Bedrock.

---

## What it does

- **Answers order questions** from the database — status, delivery dates,
  recent orders — never from the model's imagination.
- **Answers policy questions** from a retrieved corpus, with a section id
  (`[RET-1]`, `[SHP-4]`, `[WAR-2]`) attached to every claim. An answer that
  cites a section that was never retrieved is caught before the customer
  sees it.
- **Pauses for a human on refunds.** The graph freezes mid-run, the state
  is checkpointed to Postgres, and nothing resumes until someone approves
  or rejects. The process can restart in between without losing the hold.
- **Remembers across turns.** Conversations live in Postgres keyed by
  `thread_id`; long transcripts are compressed and customer profiles are
  loaded back in.
- **Screens both directions.** Prompt-injection and PII checks on the way
  in, PII sweep and citation audit on the way out.

---

## How it is put together

```
                    ┌──────────────┐
   customer  ─────► │  input_gate  │──── blocked ──────────────► END
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │    memory    │  compress transcript, load profile
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
              ┌─────┤  supervisor  ├─────┬──── direct_reply ────┐
              │     └──────▲───────┘     │                      │
              │            │             │                      │
       ┌──────▼──────┐     │      ┌──────▼───────┐              │
       │ order_agent │─────┤      │ policy_agent │              │
       └──────┬──────┘     │      └──────┬───────┘              │
              │            └─────────────┘                      │
     pending  │                                                 │
     refund   │                                    ┌────────────▼─┐
       ┌──────▼──────────┐                         │ output_rail  │──► END
       │ refund_approval │──► back to order_agent  └──────────────┘
       │  (human gate)   │
       └─────────────────┘
```

Two design rules run through the wiring:

**The guardrail sandwich.** `input_gate` is the first node and
`output_rail` is the last. Nothing reaches an agent unscreened and nothing
reaches a customer unchecked — including replies the supervisor writes
itself.

**Money decisions are data-driven, not model-driven.** The edge into the
human gate and the edge back out of it do not consult the router. The
supervisor gets no vote on whether a person is asked, and no vote on
whether a decided refund gets relayed by an agent holding the customer's
context. The refund amount comes from the order row in the database; the
model never picks a number.

### Modules

| Path | What lives there |
|---|---|
| [graph/](graph/) | State schema, graph wiring, supervisor router, memory, the refund approval gate, and the CLI |
| [agents/](agents/) | The two specialists — order lookups and policy answers |
| [tools/](tools/) | Database and retrieval tools the specialists can call |
| [guards/](guards/) | Input gate (injection + PII) and output rail (PII sweep + citation audit) |
| [rag/](rag/) | BM25 retriever over the policy corpus |
| [db/](db/) | Schema and the least-privilege role grants |
| [docs/](docs/) | The policy corpus itself — returns, shipping, warranty |
| [api/](api/) | The FastAPI service |
| [frontend/](frontend/) | The customer chat page and the staff refund-review page |
| [obs/](obs/) | Per-turn cost/latency logging and LangSmith trace config |
| [tests/](tests/) | Routing, tool-selection, groundedness, and transcript evals |

### Least privilege, at the database

Four connection strings, four roles, and none of them can do the others'
job ([db/roles.sql](db/roles.sql)):

| Role | Can do | Cannot do |
|---|---|---|
| `admin` | migrations and seeding | *(never used at runtime)* |
| `agent_ro` | `SELECT` on customers, products, orders | write anything, read refunds |
| `refund_writer` | `SELECT` + `INSERT` on refunds | `UPDATE` or `DELETE` — refund history is append-only by grant |
| `graph_writer` | read/write checkpoints and user profiles | read customers, orders, or refunds |

[seed.py](seed.py) does not just claim this — it reconnects as each role
and watches the illegal actions fail.

---

## Running it

### Prerequisites

- Python 3.13
- A Postgres database (built against Neon)
- AWS Bedrock access with Claude models enabled

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

cp .env.example .env            # then fill it in
```

The two application roles must exist with real passwords before seeding —
create them by hand in your database's SQL console. `seed.py` re-applies
their grants but deliberately will not set passwords, because that file is
committed and passwords are not.

```bash
python seed.py                  # schema + grants + fixture data + verify
python graph/build.py setup     # create the LangGraph checkpoint tables
```

### Talk to it

```bash
python graph/build.py           # CLI — shows every tool call and hop
```

```bash
uvicorn api.main:app --reload   # API + browser UI on http://localhost:8000
```

- `http://localhost:8000/` — the customer chat page
- `http://localhost:8000/review.html` — the staff refund queue
- `http://localhost:8000/docs` — live API documentation

The CLI is where you watch the machinery; the browser UI hides the steps
and shows only the answer.

### Try

```
where is my order ord_1003?
how long do I have to return it?        <- tests memory across turns
my headphones arrived broken, can I get a refund?
```

The last one parks a proposal at the human gate. Approve it at the CLI
prompt, or from the review page in another tab.

---

## Configuration

All settings come from the environment; see [.env.example](.env.example)
for the full list.

| Variable | Purpose |
|---|---|
| `AWS_BEARER_TOKEN_BEDROCK`, `AWS_REGION` | Bedrock access |
| `SHOPSENSE_DB_URL_ADMIN` / `_RO` / `_WRITER` / `_GRAPH` | one per database role |
| `SHOPSENSE_MODEL_AGENT` | specialist model (default: Sonnet 4.6) |
| `SHOPSENSE_MODEL_ROUTER` | routing model (default: Haiku 4.5) |
| `SHOPSENSE_MAX_TOKENS`, `SHOPSENSE_EFFORT` | generation limits |
| `SHOPSENSE_GUARDRAILS_MODE` | `monitor` (log only) or `enforce` (block) |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | optional tracing |

Pricing tables and per-model capability flags live in
[config.py](config.py), which is what makes the cost footer a real number
rather than an estimate.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness. Touches no dependency, on purpose |
| `POST` | `/chat` | Send a message on a `thread_id` |
| `POST` | `/approve` | Resolve a pending refund and resume the run |
| `GET` | `/conversations/{thread_id}` | Full transcript |
| `GET` | `/conversations/{thread_id}/state` | Is this conversation frozen, and on what? |
| `POST` | `/conversations/{thread_id}/close` | End a conversation |
| `GET` | `/policies` | The policy corpus, by section |
| `GET` | `/metrics` | Turns, cost, and p50/p95 latency across logged runs |

The server keeps **no** conversation in memory. Every request carries a
`thread_id` and the history lives in Postgres, which is what lets you run
several copies behind a load balancer.

---

## Tests

```bash
pytest                       # everything the environment allows
pytest -m "not live"         # no credentials, no database, ~3 seconds
pytest -m live               # calls Bedrock; costs a few cents
```

Live tests **skip** rather than fail when credentials are absent, so the
suite is green on a fresh checkout with no `.env`.

What is actually asserted: that the router sends each question to the right
specialist and cannot invent a route; that the hop cap holds; that a
decided refund is always relayed by an agent; that retrieval ranks the
right policy section; that a cited answer's citations exist; and that state
messages are never mutated in place.

---

## CI/CD

[.github/workflows/ci.yml](.github/workflows/ci.yml) runs four jobs:

1. **test** — credential-free evals. Runs for everyone, including forks.
2. **live-evals** — Bedrock + database evals, gated behind `test` so a red
   suite never costs money.
3. **docker** — proves the image still builds, without pushing it.
4. **deploy** — builds, pushes to ECR, and triggers a rolling ECS
   deployment. `main` only, and only when `AWS_DEPLOY_ENABLED=true`.

`deploy` *needs* the other three, so a broken router makes deployment
impossible rather than merely inadvisable. AWS credentials come from OIDC —
there are no long-lived keys in the repository. Images carry two tags: the
git sha (immutable, so "what is in production?" has an answer) and
`latest` (what the service pulls).

The container runs as a non-root user and ships a `HEALTHCHECK`. ECS shifts
traffic only after the new task passes it, so a bad build degrades to
"no deploy" rather than "outage".

---

## Observability

Every turn appends a line to `runs/cost.jsonl` with token counts, cost,
latency, and which nodes ran — timing is wrapped around every node, so a
node added tomorrow is instrumented without anyone remembering to.
`GET /metrics` aggregates it. Guardrail hits are logged separately to
`runs/gate_triggers.jsonl`, with the offending text masked.

Set `LANGSMITH_TRACING=true` for full run traces, tagged by thread and
customer.
