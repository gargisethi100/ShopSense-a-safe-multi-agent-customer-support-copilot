"""Build the demo world: schema + grants + deterministic data, then verify.

WHAT "SEEDING" MEANS
    A brand-new database is an empty filing room. Seeding fills it with
    known demo data - our fake customers, products, and orders - so there
    is something to point the agent at.

WHY THE DATA IS HARD-CODED (DETERMINISTIC), NOT RANDOM
    Because the eval suite (Phase 10) will assert things like "when asked
    about ord_1003, the agent reports it was delivered on Jul 23". That
    test is only stable if ord_1003 IS ALWAYS the same order. Random or
    faker-generated data would make every test flaky by construction.
    Rule: demo data is part of the test fixture, so it must be as fixed
    as the tests themselves.

WHY WIPE-AND-REBUILD EVERY RUN
    This script starts by re-running schema.sql, which DROPs and recreates
    all tables. For a demo/eval database that is a feature: no matter what
    state previous experiments left behind, one command returns the world
    to a known-good snapshot. (You would obviously never do this to a
    production database - the difference is that THIS data is a fixture,
    not a record.)

ORDER OF OPERATIONS (and why the order is load-bearing)
    1. schema.sql   - rebuild tables      (DROPs destroy existing grants!)
    2. roles.sql    - re-apply grants     (must follow every rebuild)
    3. insert data  - as admin
    4. verify       - reconnect as agent_ro / refund_writer and PROVE the
                      permissions hold. A security claim you haven't watched
                      fail an illegal action is just a claim.

PREREQUISITES (one-time, done by hand in Neon's SQL editor):
    The two roles must already exist WITH REAL PASSWORDS. This script
    re-applies their grants but deliberately cannot create their passwords:
    passwords don't belong in a committed file, and this file is committed.
    If the roles are missing, the script stops and tells you what to do.

Run it:
    python seed.py
"""

from __future__ import annotations

from pathlib import Path

import psycopg

from db.pool import admin_connection, get_ro_pool, get_writer_pool

DB_DIR = Path(__file__).parent / "db"

# ---------------------------------------------------------------------------
# The fixture. Every id and date here may be referenced by evals in Phase 10,
# by the README's demo script, and by you while debugging. Change values only
# on purpose, never casually - a changed fixture is a changed test suite.
# ---------------------------------------------------------------------------

CUSTOMERS = [
    # (id, name, email, phone)
    ("cust_001", "Dana Lee", "dana@example.com", "+1-555-0101"),
    ("cust_002", "Sam Green", "sam@example.com", None),
    ("cust_003", "Priya Patel", "priya@example.com", "+1-555-0103"),
]

PRODUCTS = [
    # (id, sku, name, category, price_usd)
    ("prod_001", "SKU-HP-01", "Aurora Wireless Headphones", "electronics", 79.99),
    ("prod_002", "SKU-HD-02", "Cloud Fleece Hoodie", "apparel", 39.99),
    ("prod_003", "SKU-MG-03", "Terracotta Coffee Mug", "home", 12.50),
    ("prod_004", "SKU-KB-04", "Quiet-Click Keyboard", "electronics", 89.00),
    ("prod_005", "SKU-WB-05", "Alpine Steel Water Bottle", "outdoors", 18.75),
]

ORDERS = [
    # (id, customer_id, product_id, qty, total_usd, status, tracking_no,
    #  ordered_at, delivered_at)
    # Every status from schema.sql's CHECK list appears at least once, so
    # every tool code path has a row to exercise it.
    ("ord_1001", "cust_001", "prod_002", 1, 39.99, "delivered", "TRK-11007",
     "2026-07-05 10:15:00+00", "2026-07-09 16:40:00+00"),
    ("ord_1002", "cust_002", "prod_003", 2, 25.00, "shipped", "TRK-84213",
     "2026-07-24 08:02:00+00", None),
    # ord_1003 is THE demo order: delivered headphones for Dana, the one the
    # README's refund walkthrough uses. Treat it as sacred.
    ("ord_1003", "cust_001", "prod_001", 1, 79.99, "delivered", "TRK-52990",
     "2026-07-18 14:30:00+00", "2026-07-23 11:05:00+00"),
    ("ord_1004", "cust_003", "prod_004", 1, 89.00, "processing", None,
     "2026-08-01 19:45:00+00", None),
    ("ord_1005", "cust_002", "prod_001", 1, 79.99, "cancelled", None,
     "2026-06-20 12:00:00+00", None),
    ("ord_1006", "cust_003", "prod_005", 3, 56.25, "shipped", "TRK-99310",
     "2026-07-30 09:20:00+00", None),
]


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def roles_exist(conn: psycopg.Connection) -> bool:
    """True if both runtime roles are present in the server's role catalog."""
    rows = conn.execute(
        "SELECT rolname FROM pg_roles WHERE rolname IN ('agent_ro', 'refund_writer')"
    ).fetchall()
    return len(rows) == 2


def apply_sql_file(conn: psycopg.Connection, path: Path) -> None:
    """Run a whole .sql file as one batch.

    psycopg allows multi-statement strings when no parameters are passed,
    which is exactly the schema/grants case. Data inserts below do NOT use
    this - they use parameters, for the reason explained there.
    """
    conn.execute(path.read_text(encoding="utf-8"))


def insert_fixture(conn: psycopg.Connection) -> None:
    """Insert the demo rows, the safe way.

    Note the %s placeholders: the SQL text and the VALUES travel to Postgres
    SEPARATELY, and the driver guarantees a value can never be mis-read as
    SQL. This is 'parameterized SQL' - the standard defence against SQL
    injection. It matters here as a habit: Phase 2's tools will run queries
    containing STRINGS PRODUCED BY AN LLM, and there it is non-negotiable.
    Building the habit starts in the safest file, not the scariest one.
    """
    for row in CUSTOMERS:
        conn.execute(
            "INSERT INTO customers (id, name, email, phone) VALUES (%s, %s, %s, %s)",
            row,
        )
    for row in PRODUCTS:
        conn.execute(
            "INSERT INTO products (id, sku, name, category, price_usd)"
            " VALUES (%s, %s, %s, %s, %s)",
            row,
        )
    for row in ORDERS:
        conn.execute(
            "INSERT INTO orders (id, customer_id, product_id, quantity, total_usd,"
            " status, tracking_no, ordered_at, delivered_at)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            row,
        )


def verify_least_privilege() -> None:
    """Reconnect as the two WEAK roles and prove their limits are real.

    This is the phase-closing test: not "did the inserts work" (we saw
    that) but "does the security design hold when exercised". Every
    permission-denied below is a passing assertion.
    """
    print("\nverification (connecting as the runtime roles)")
    print("-" * 46)

    with get_ro_pool().connection() as conn:
        n = conn.execute("SELECT count(*) FROM orders").fetchone()[0]
        print(f"  agent_ro       SELECT orders        : ok ({n} rows)")
        try:
            conn.execute("UPDATE orders SET status = 'delivered'")
            raise SystemExit("  agent_ro could WRITE - roles.sql was not applied!")
        except psycopg.errors.InsufficientPrivilege:
            print("  agent_ro       UPDATE orders        : permission denied (correct)")

    with get_writer_pool().connection() as conn:
        try:
            conn.execute("SELECT count(*) FROM customers")
            raise SystemExit("  refund_writer could READ - roles.sql was not applied!")
        except psycopg.errors.InsufficientPrivilege:
            print("  refund_writer  SELECT customers     : permission denied (correct)")

    # Prove the writer CAN insert - without leaving a fake refund behind.
    # The writer pool is transactional (autocommit=False), so we insert and
    # then ROLL BACK: Postgres checks every permission and constraint as if
    # for real, then un-happens the row. A dress rehearsal with real props.
    with get_writer_pool().connection() as conn:
        conn.execute(
            "INSERT INTO refunds (id, order_id, amount_usd, reason, approved_by)"
            " VALUES (%s, %s, %s, %s, %s)",
            ("ref_smoke_test", "ord_1003", 79.99, "seed.py rehearsal", "seed-script"),
        )
        conn.rollback()
        print("  refund_writer  INSERT refunds       : ok (rolled back, no row kept)")


def main() -> None:
    try:
        with admin_connection() as conn:
            if not roles_exist(conn):
                print(
                    "Roles agent_ro / refund_writer do not exist yet.\n"
                    "  1. Open Neon -> SQL Editor\n"
                    "  2. Paste db/roles.sql, replace the two CHANGE_ME passwords\n"
                    "  3. Run it, then put the matching URLs in .env\n"
                    "This script re-applies grants but will not invent passwords:\n"
                    "a committed file must never contain one."
                )
                raise SystemExit(1)

            print("rebuilding schema (schema.sql)...")
            apply_sql_file(conn, DB_DIR / "schema.sql")

            # DROP TABLE just destroyed all grants; restore them immediately.
            # (Existing roles are untouched - the DO block skips creation -
            # so the real passwords you set in Neon survive this.)
            print("re-applying grants (roles.sql)...")
            apply_sql_file(conn, DB_DIR / "roles.sql")

            print("inserting fixture data...")
            insert_fixture(conn)

            for table in ("customers", "products", "orders"):
                n = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # noqa: S608
                print(f"  {table:9} : {n} rows")
    except RuntimeError as e:
        # Missing .env URL: require_db_url already wrote the instructions.
        print(f"\n{e}")
        raise SystemExit(0)

    verify_least_privilege()

    print("\nSeed complete. The demo world exists and the keycards hold.")
    print("Phase 1 (data layer) is DONE.")


if __name__ == "__main__":
    main()
