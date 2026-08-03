-- ===========================================================================
-- ShopSense roles — the AI's locked-down database identities.
--
-- BIG IDEA (read this first):
--   A Postgres "role" is a login identity — a username/password — that carries
--   a list of permissions. Permissions are handed out with GRANT and taken
--   away with REVOKE, per table, per action (SELECT / INSERT / UPDATE / ...).
--
--   We create two deliberately weak identities:
--
--     agent_ro       can SELECT (read) customers, products, orders. NOTHING else.
--     refund_writer  can INSERT (append) into refunds. NOTHING else.
--
--   The AI's tools will connect as these roles. So even if a user tricks the
--   model into attempting "DROP TABLE orders" or "UPDATE orders SET ...",
--   Postgres itself answers: `ERROR: permission denied`. The safety property
--   does not depend on the model behaving; it is enforced by the database.
--   That is the difference between DISCOURAGED (a rule in a prompt) and
--   IMPOSSIBLE (a rule in the infrastructure).
--
-- BEFORE RUNNING — passwords:
--   This file is committed to git, so it must NEVER contain a real password.
--   The two CHANGE_ME placeholders below are intentionally invalid. Replace
--   them (in the Neon SQL editor, NOT in this file) with strong passwords,
--   e.g. from PowerShell:   -join ((48..57)+(97..122) | Get-Random -Count 24 | % {[char]$_})
--
-- HOW TO RUN (Neon):
--   1. neon.tech -> your project -> "SQL Editor" (left sidebar).
--      The editor runs as your project's owner role — the same identity as
--      SHOPSENSE_DB_URL_ADMIN — which is what makes it allowed to create roles.
--   2. Make sure schema.sql has been run first (the tables must exist,
--      because grants attach to tables).
--   3. Paste this whole file, swap the two CHANGE_ME strings, hit Run.
--
-- RE-RUNNABLE ON PURPOSE, and it MUST be re-run after every schema rebuild:
--   a GRANT lives on the table object itself. schema.sql starts with
--   DROP TABLE, and dropping a table destroys its grants with it. So the
--   rule is: schema.sql, then roles.sql, always as a pair. (seed.py will
--   automate exactly that order in the next file.)
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- 1. Create the two roles (only if they don't already exist).
--
--    CREATE ROLE has no "IF NOT EXISTS" in Postgres, so we wrap it in a DO
--    block — a tiny inline script — that checks the role catalog first.
--    This is what makes the file safe to run twice.
--
--    LOGIN     = this role may open connections (it's a real username).
--    PASSWORD  = how it authenticates. Placeholders MUST be replaced.
-- ---------------------------------------------------------------------------

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'agent_ro') THEN
        CREATE ROLE agent_ro LOGIN PASSWORD 'CHANGE_ME_RO';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'refund_writer') THEN
        CREATE ROLE refund_writer LOGIN PASSWORD 'CHANGE_ME_WRITER';
    END IF;
END
$$;

-- ---------------------------------------------------------------------------
-- 2. Start from zero: revoke everything, then grant back the minimum.
--
--    WHY REVOKE FIRST? Two reasons:
--    a) Postgres ships with some default openness (e.g. the PUBLIC
--       pseudo-role historically could CREATE in the public schema).
--       We don't want to memorize which defaults exist on which version —
--       we set the floor to zero explicitly and build up from there.
--    b) It makes this file CONVERGENT: whatever mess previous experiments
--       left behind, running this file always lands on exactly the grants
--       written below. The file is the single source of truth you can audit.
-- ---------------------------------------------------------------------------

REVOKE ALL ON ALL TABLES IN SCHEMA public FROM agent_ro, refund_writer;
REVOKE ALL ON SCHEMA public FROM agent_ro, refund_writer;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;  -- nobody creates tables by accident

-- ---------------------------------------------------------------------------
-- 3. The grants. This short block IS the security policy — read it as prose.
--
--    USAGE on the schema = "may walk into the room where the tables live".
--    Without it, a role can't reach any table no matter what else we grant.
--    It does NOT by itself allow reading anything.
-- ---------------------------------------------------------------------------

GRANT USAGE ON SCHEMA public TO agent_ro, refund_writer;

-- The read path: exactly the three tables the specialists may look at.
-- Note what is ABSENT: no INSERT, UPDATE, DELETE, TRUNCATE — and no grant at
-- all on refunds or user_profiles. agent_ro cannot even see refund history.
GRANT SELECT ON customers, products, orders TO agent_ro;

-- The single write path: append one row per approved refund.
--   no UPDATE  -> a refund row can never be edited after the fact
--   no DELETE  -> ...or erased. Financial history is append-only BY GRANT.
--   SELECT     -> added in Phase 2: the refund tool must answer "was this
--                 order ALREADY refunded?" before proposing a new one, and
--                 refunds history is (correctly) invisible to agent_ro.
--                 This is how grants should evolve: a permission appears
--                 the day a feature needs it, with the reason written next
--                 to it - never "just in case".
-- (The order_id foreign-key check still works: Postgres validates FKs
--  internally as the table owner, so the writer needs no grant on orders.)
GRANT SELECT, INSERT ON refunds TO refund_writer;

-- Nothing to grant on sequences: our IDs are TEXT ('cust_001', 'ref_...'),
-- not auto-increment counters, so there are no sequence objects to expose.
-- (With SERIAL ids, INSERT would also need GRANT USAGE ON the sequence —
--  a classic "why is my insert failing" trap you get to skip.)

-- ---------------------------------------------------------------------------
-- Deliberately NOT granted yet (each arrives with the phase that needs it):
--
--   user_profiles  -> Phase 6 (memory). The profile reader/writer grant is
--                     added when we build that feature, not before. Grants
--                     should trace to a feature that exists.
--
--   LangGraph checkpoint tables -> Phase 4. PostgresSaver creates and owns
--                     its own tables; we'll decide its connection identity
--                     when we wire the graph.
-- ---------------------------------------------------------------------------

-- ---------------------------------------------------------------------------
-- 4. Prove it worked (paste these AFTER running the above, one at a time).
--
--    SET ROLE temporarily "becomes" another role inside your admin session —
--    a way to test permissions without opening a new connection.
-- ---------------------------------------------------------------------------
--
--   SET ROLE agent_ro;
--   SELECT count(*) FROM customers;      -- works (0 rows for now — fine)
--   DELETE FROM orders;                  -- ERROR: permission denied  <- the win
--   INSERT INTO refunds VALUES ('x','x',1,'x','x');  -- ERROR: permission denied
--   RESET ROLE;
--
--   SET ROLE refund_writer;
--   SELECT * FROM customers;             -- ERROR: permission denied
--   RESET ROLE;
--
-- Every ERROR above is a success: it is the database saying no.
