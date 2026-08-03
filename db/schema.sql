-- ===========================================================================
-- ShopSense schema — the shape of the data, as plain SQL.
--
-- WHY A .sql FILE AND NOT AN ORM?
--   An ORM earns its keep when application code creates/updates rows all day.
--   Here, the application READS through three tables and performs exactly one
--   kind of write (a refund). The schema is the security boundary — roles.sql
--   grants privileges table-by-table — and you cannot reason about grants on
--   tables an ORM conjures at import time. SQL you can read is SQL you can
--   audit.
--
-- WHO RUNS THIS: seed.py, over the ADMIN connection (SHOPSENSE_DB_URL_ADMIN).
--   The runtime agent roles never have permission to run DDL like this.
--
-- RE-RUNNABLE ON PURPOSE: `DROP ... IF EXISTS ... CASCADE` at the top means
--   running it twice gives you the same database, not an error. For a demo
--   with deterministic seed data, "wipe and rebuild" is a feature: every eval
--   run starts from a known world.
--
-- WHAT IS DELIBERATELY NOT HERE:
--   * LangGraph's checkpoint tables (Phase 4). PostgresSaver.setup() creates
--     and owns those itself; hand-writing them would break on library upgrade.
--   * Indexes beyond primary/foreign keys. The demo tables hold dozens of
--     rows; an index catalogue would be cargo cult. Add them when a query is
--     measurably slow, not before.
-- ===========================================================================

-- Order matters twice in this file:
--   drops: children before parents (or CASCADE handles it, but be explicit),
--   creates: parents before children, because FKs must reference something.
DROP TABLE IF EXISTS refunds CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS user_profiles CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- ---------------------------------------------------------------------------
-- The three tables agent_ro may SELECT from (grants live in roles.sql)
-- ---------------------------------------------------------------------------

CREATE TABLE customers (
    -- TEXT ids like 'cust_001' instead of SERIAL integers. Two reasons:
    --  1. Deterministic: seed.py can hard-code them, so evals can hard-code
    --     them too ("cust_001 asks about order ord_1003" is a stable test).
    --  2. Self-describing in a trace: when an agent passes 'cust_001' to a
    --     tool you know instantly what kind of thing it is; a bare `7` could
    --     be anything.
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    -- Email is the natural lookup key ("hi, it's dana@example.com") — UNIQUE
    -- makes find_customer(email) return at most one row BY CONSTRUCTION,
    -- so the tool never needs a "which of these did you mean?" branch.
    email       TEXT NOT NULL UNIQUE,
    phone       TEXT,                       -- nullable: not everyone gives one
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE products (
    id          TEXT PRIMARY KEY,           -- 'prod_001'
    sku         TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL,
    -- NUMERIC, never FLOAT, for money. FLOAT cannot represent 19.99 exactly;
    -- the error is invisible until a refund total is off by one cent and a
    -- human asks why. NUMERIC(10,2) stores exact cents up to $99,999,999.99.
    price_usd   NUMERIC(10, 2) NOT NULL CHECK (price_usd >= 0),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE orders (
    id           TEXT PRIMARY KEY,          -- 'ord_1001'
    -- REFERENCES = the database refuses an order pointing at a customer that
    -- doesn't exist. Tools then never have to handle the "order with a
    -- dangling customer" case, because that state is unrepresentable.
    customer_id  TEXT NOT NULL REFERENCES customers(id),
    product_id   TEXT NOT NULL REFERENCES products(id),
    quantity     INTEGER NOT NULL CHECK (quantity > 0),
    -- Denormalised on purpose: the price AT ORDER TIME. If products.price_usd
    -- changes next week, historical orders (and refund amounts!) must not
    -- silently change with it. Point-in-time facts get copied, not joined.
    total_usd    NUMERIC(10, 2) NOT NULL CHECK (total_usd >= 0),
    -- TEXT + CHECK instead of a Postgres ENUM type. Same integrity guarantee,
    -- but adding a status later is `ALTER TABLE ... DROP/ADD CONSTRAINT`,
    -- not the special ceremony ALTER TYPE needs. The CHECK also doubles as
    -- documentation: this line IS the list of legal states.
    status       TEXT NOT NULL CHECK (status IN
                     ('processing', 'shipped', 'delivered', 'cancelled')),
    -- Nullable columns that are only meaningful in some states. NULL here
    -- means "not applicable yet", and tools must say so in words
    -- ("not yet shipped"), never leak a raw None to the model.
    tracking_no  TEXT,
    ordered_at   TIMESTAMPTZ NOT NULL,      -- set by seed.py, not now(): demo
                                            -- orders need realistic past dates
    delivered_at TIMESTAMPTZ
);

-- ---------------------------------------------------------------------------
-- The ONE table refund_writer may INSERT into
-- ---------------------------------------------------------------------------

CREATE TABLE refunds (
    -- The id arrives from the application ('ref_' + uuid), not a sequence.
    -- Generating it BEFORE the interrupt() pause means the approval shown to
    -- the human and the row that lands in the table share an id you can grep
    -- for across the UI, the trace, and the database.
    id           TEXT PRIMARY KEY,
    order_id     TEXT NOT NULL REFERENCES orders(id),
    amount_usd   NUMERIC(10, 2) NOT NULL CHECK (amount_usd > 0),
    reason       TEXT NOT NULL,
    -- Who clicked Approve. NOT NULL is the schema stating a policy: a refund
    -- row without a human approver CANNOT EXIST. The graph pauses at
    -- interrupt() precisely to obtain this value.
    approved_by  TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- Append-only by GRANT, not by trigger: roles.sql gives refund_writer INSERT
-- only — no UPDATE, no DELETE. A refund is a financial event; if one was
-- wrong, the fix is a compensating entry, never an edit. (This is why there
-- is no `status` column: a row's existence means it happened.)

-- ---------------------------------------------------------------------------
-- Phase 6 memory store (created now so roles.sql can grant on it in one pass)
-- ---------------------------------------------------------------------------

CREATE TABLE user_profiles (
    -- One row per customer, upserted after each session. PRIMARY KEY on
    -- customer_id makes `INSERT ... ON CONFLICT (customer_id) DO UPDATE`
    -- (the upsert in Phase 6) work with no extra index.
    customer_id  TEXT PRIMARY KEY REFERENCES customers(id),
    -- JSONB, not columns-per-fact: the LLM distills an open-ended profile
    -- ("prefers email", "has a standing warranty issue") and we cannot
    -- enumerate its keys in advance. JSONB is Postgres saying "schemaless
    -- HERE, and only here" — the escape hatch is fenced to one column.
    profile      JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
