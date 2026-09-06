"""Parse a Letterboxd CSV export (diary.csv and/or ratings.csv) into SQLite.

Usage:
    python ingest.py --data-dir data/
    python ingest.py --diary data/diary.csv --ratings data/ratings.csv

Dedupes by (title, year). Rows are processed in ascending date order so that,
for a given film, the most recently dated row wins for watched_date; a row's
rating only overwrites the running value when that row actually has one, so a
later unrated rewatch entry doesn't blank out a previously known rating.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from db import get_conn, init_db


@dataclass(frozen=True, slots=True)
class CsvRow:
    """A single normalized diary.csv / ratings.csv row."""

    sort_date: datetime | None
    title: str
    year: int | None
    rating: float | None
    watched_date: str | None


@dataclass(slots=True)
class MovieRecord:
    """Accumulator for a (title, year) key while merging rows across files."""

    title: str
    year: int | None
    rating: float | None = None
    watched_date: str | None = None

    def absorb(self, row: CsvRow) -> None:
        """Apply a row's fields, newest-row-wins, without blanking known
        values from a subsequent row that lacks them."""
        if row.rating is not None:
            self.rating = row.rating
        if row.watched_date is not None:
            self.watched_date = row.watched_date


def _parse_date(value: str | None) -> datetime | None:
    """A "YYYY-MM-DD" string as a datetime, or None if blank/unparseable."""
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d")
    except ValueError:
        return None


def _parse_rating(value: str | None) -> float | None:
    """A Letterboxd star-rating string as a float, or None if blank/invalid."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_year(value: str | None) -> int | None:
    """A release-year string as an int, or None if blank/invalid."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _read_rows(path: Path, watched_date_col: str) -> list[CsvRow]:
    """Every row of a diary.csv or ratings.csv export as a CsvRow."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            title = (raw.get("Name") or "").strip()
            if not title:
                continue
            year = _parse_year(raw.get("Year"))
            rating = _parse_rating(raw.get("Rating"))
            watched_date = (raw.get(watched_date_col) or "").strip() or None
            # Sort key: prefer the watched-date column, fall back to the
            # generic "Date" column (diary entry / rating log date).
            sort_date = _parse_date(watched_date) or _parse_date(raw.get("Date"))
            rows.append(CsvRow(sort_date, title, year, rating, watched_date))
    return rows


def ingest(diary_path: Path | None, ratings_path: Path | None) -> int:
    """Merge and upsert diary.csv and/or ratings.csv into the movies table.

    Returns the number of movies ingested/updated.
    """
    rows: list[CsvRow] = []
    if diary_path and diary_path.exists():
        rows.extend(_read_rows(diary_path, watched_date_col="Watched Date"))
        print(f"Read {diary_path} ({sum(1 for _ in open(diary_path)) - 1} rows)")
    if ratings_path and ratings_path.exists():
        rows.extend(_read_rows(ratings_path, watched_date_col="Date"))
        print(f"Read {ratings_path} ({sum(1 for _ in open(ratings_path)) - 1} rows)")

    if not rows:
        print("No input rows found. Nothing to ingest.", file=sys.stderr)
        return 0

    # Process oldest -> newest so the last write for a given (title, year)
    # reflects the most recent watch/rating.
    rows.sort(key=lambda r: r.sort_date or datetime.min)

    merged: dict[tuple[str, int | None], MovieRecord] = {}
    for row in rows:
        key = (row.title.lower(), row.year)
        record = merged.setdefault(key, MovieRecord(title=row.title, year=row.year))
        record.absorb(row)

    init_db()
    conn = get_conn()
    try:
        for record in merged.values():
            conn.execute(
                """
                INSERT INTO movies (title, year, letterboxd_rating, watched_date)
                VALUES (:title, :year, :rating, :watched_date)
                ON CONFLICT(title, year) DO UPDATE SET
                    letterboxd_rating = excluded.letterboxd_rating,
                    watched_date = excluded.watched_date
                """,
                {
                    "title": record.title,
                    "year": record.year,
                    "rating": record.rating,
                    "watched_date": record.watched_date,
                },
            )
        conn.commit()
    finally:
        conn.close()

    print(f"Ingested/updated {len(merged)} movies.")
    return len(merged)


def ingest_from_data_dir(data_dir: Path | None = None) -> int:
    """Ingest diary.csv and/or ratings.csv from `data_dir` (default: the
    `data/` directory next to this file), whichever of the two exist.

    Shared by both `main()` below and the app's "Sync from data/" button
    (see POST /sync in app.py), so there's one place that resolves the
    default paths and decides what "nothing to ingest" means.

    Raises FileNotFoundError if neither file exists.
    """
    data_dir = data_dir or (Path(__file__).parent / "data")
    diary_path = data_dir / "diary.csv"
    ratings_path = data_dir / "ratings.csv"
    if not diary_path.exists() and not ratings_path.exists():
        raise FileNotFoundError(f"Neither {diary_path} nor {ratings_path} exists.")
    return ingest(diary_path, ratings_path)


def main() -> None:
    """Parse CLI args and run `ingest` against the resolved CSV paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing diary.csv and/or ratings.csv (default names assumed).",
    )
    parser.add_argument("--diary", type=Path, default=None, help="Path to diary.csv")
    parser.add_argument("--ratings", type=Path, default=None, help="Path to ratings.csv")
    args = parser.parse_args()

    diary_path: Path | None = args.diary
    ratings_path: Path | None = args.ratings
    if args.data_dir:
        diary_path = diary_path or args.data_dir / "diary.csv"
        ratings_path = ratings_path or args.data_dir / "ratings.csv"
    if diary_path is None and ratings_path is None:
        try:
            ingest_from_data_dir()
        except FileNotFoundError as e:
            print(f"{e} Pass --data-dir, --diary, or --ratings.", file=sys.stderr)
            sys.exit(1)
        return

    if not (diary_path and diary_path.exists()) and not (
        ratings_path and ratings_path.exists()
    ):
        print(
            f"Neither {diary_path} nor {ratings_path} exists. "
            "Pass --data-dir, --diary, or --ratings.",
            file=sys.stderr,
        )
        sys.exit(1)

    ingest(diary_path, ratings_path)


if __name__ == "__main__":
    main()
