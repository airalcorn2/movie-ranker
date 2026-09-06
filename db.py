"""SQLite schema and connection helper shared by ingest.py, model.py, and app.py."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "movie_ranker.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS movies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    year INTEGER,
    letterboxd_rating REAL,
    watched_date TEXT,
    poster_url TEXT,
    UNIQUE(title, year)
);

CREATE TABLE IF NOT EXISTS comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_a_id INTEGER NOT NULL REFERENCES movies(id),
    movie_b_id INTEGER NOT NULL REFERENCES movies(id),
    winner_id INTEGER REFERENCES movies(id),
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'user'
);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    n_comparisons INTEGER NOT NULL,
    scores_json TEXT NOT NULL,
    kendall_tau_vs_prev REAL
);
"""


def get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """A new connection with row access by column name and FKs enforced."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the schema if it doesn't already exist, and apply any pending
    migrations to an existing database created before a schema change."""
    conn = get_conn(db_path)
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.commit()
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an existing `comparisons` table up to the current schema.

    `CREATE TABLE IF NOT EXISTS` above only affects brand-new databases, so
    a database created before a column was added needs it backfilled here.
    ALTER TABLE ... ADD COLUMN is a fast, additive, non-destructive
    operation in SQLite -- it doesn't rewrite or touch existing row data.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(comparisons)")}
    if "source" not in columns:
        # Existing rows predate the concept of a comparison source, and were
        # all real clicks, so 'user' is the correct backfilled value for them.
        conn.execute("ALTER TABLE comparisons ADD COLUMN source TEXT NOT NULL DEFAULT 'user'")


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DB_PATH}")
