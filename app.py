"""FastAPI server: /, /status, /next_pair, /compare, /comparisons, /movies,
/sync, /refit, /seed_rating_comparisons, /export, /export_comparisons."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

import model
from db import get_conn, init_db
from ingest import ingest_from_data_dir
from posters import fetch_poster_url

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the SQLite schema on startup."""
    init_db()
    yield


app = FastAPI(title="Letterboxd Ranker", lifespan=lifespan)


# -- Response / request schemas -------------------------------------------


class MovieOut(BaseModel):
    """A movie with everything the comparison UI needs to display it."""

    id: int
    title: str
    year: int | None
    letterboxd_rating: float | None
    poster_url: str | None


class PairOut(BaseModel):
    """A pair of movies to compare next, plus how they were chosen."""

    a: MovieOut
    b: MovieOut
    strategy: model.SelectionStrategy


class RankedMovieOut(BaseModel):
    """A single row of the live top-20 ranking."""

    id: int
    title: str
    year: int | None
    score: float
    n_comparisons: int


class StatusOut(BaseModel):
    """Overall progress: counts, connectivity, stability, and the top-20."""

    n_movies: int
    n_comparisons_decisive: int  # Yours only -- excludes rating-derived comparisons.
    n_comparisons_total: int  # Yours only (decisive + skipped), same exclusion.
    n_rating_derived: int  # From star ratings -- see POST /seed_rating_comparisons.
    n_components: int
    is_connected: bool
    kendall_tau_vs_prev: float | None
    last_fit_timestamp: str | None
    top20: list[RankedMovieOut]


class CompareRequest(BaseModel):
    """A single comparison result to record."""

    movie_a_id: int
    movie_b_id: int
    winner_id: int | None = None  # None means skip / can't decide.


class CompareResult(BaseModel):
    """Where the comparison count stands after recording or editing one."""

    n_comparisons_decisive: int


class RefitResult(BaseModel):
    """The outcome of an explicit POST /refit."""

    n_comparisons_decisive: int
    kendall_tau_vs_prev: float | None
    elapsed_seconds: float


class SeedResult(BaseModel):
    """The outcome of an explicit POST /seed_rating_comparisons."""

    n_added: int
    n_comparisons_total: int


class SyncResult(BaseModel):
    """The outcome of an explicit POST /sync."""

    n_movies_ingested: int
    n_movies_total: int


class MovieRefOut(BaseModel):
    """A minimal movie reference, for embedding in a ComparisonOut."""

    id: int
    title: str
    year: int | None


class ComparisonOut(BaseModel):
    """A single row in the editable comparison history."""

    id: int
    movie_a: MovieRefOut
    movie_b: MovieRefOut
    winner_id: int | None
    timestamp: str
    source: model.ComparisonSource


class ComparisonListOut(BaseModel):
    """One page of comparison history."""

    comparisons: list[ComparisonOut]
    total: int  # Total matching rows (before pagination), for a "N of M" display.


class MovieListOut(BaseModel):
    """All movies, for the history filter's datalist."""

    movies: list[MovieRefOut]


class UpdateComparisonRequest(BaseModel):
    """A correction to an existing comparison's winner."""

    winner_id: int | None  # Must be movie_a_id, movie_b_id, or None for skip.


# -- Helpers ----------------------------------------------------------------


def _to_movie_out(
    movie: model.Movie, conn: sqlite3.Connection, fetch_missing_poster: bool = False
) -> MovieOut:
    """A Movie as a MovieOut, optionally fetching and caching its poster."""
    poster_url = movie.poster_url
    if poster_url is None and fetch_missing_poster:
        poster_url = fetch_poster_url(movie.title, movie.year)
        if poster_url is not None:
            conn.execute(
                "UPDATE movies SET poster_url = ? WHERE id = ?", (poster_url, movie.id)
            )
            conn.commit()
    return MovieOut(
        id=movie.id,
        title=movie.title,
        year=movie.year,
        letterboxd_rating=movie.letterboxd_rating,
        poster_url=poster_url,
    )


def _to_ranked_movie_out(
    movie: model.Movie, scores: model.ScoreMap, counts: dict[model.MovieId, int]
) -> RankedMovieOut:
    """A Movie as a RankedMovieOut, with its current score and comparison count."""
    return RankedMovieOut(
        id=movie.id,
        title=movie.title,
        year=movie.year,
        score=round(scores.get(movie.id, 0.0), 3),
        n_comparisons=counts.get(movie.id, 0),
    )


def _to_comparison_out(
    record: model.ComparisonRecord, movies_by_id: dict[model.MovieId, model.Movie]
) -> ComparisonOut | None:
    """A ComparisonRecord as a ComparisonOut, with its movies' titles/years
    resolved in."""
    movie_a = movies_by_id.get(record.movie_a_id)
    movie_b = movies_by_id.get(record.movie_b_id)
    if movie_a is None or movie_b is None:
        return None  # Orphaned row (shouldn't happen); skip rather than 500.
    return ComparisonOut(
        id=record.id,
        movie_a=MovieRefOut(id=movie_a.id, title=movie_a.title, year=movie_a.year),
        movie_b=MovieRefOut(id=movie_b.id, title=movie_b.title, year=movie_b.year),
        winner_id=record.winner_id,
        timestamp=record.timestamp,
        source=record.source,
    )


# -- Routes -------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    """Serve the comparison UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/status", response_model=StatusOut)
def status() -> StatusOut:
    """Overall progress: counts, connectivity, stability, and the top-20."""
    conn = get_conn()
    try:
        movies = model.get_movies(conn)
        components = model.connected_components(conn, movies) if movies else []

        run = model.latest_model_run(conn)
        scores = model.current_scores(conn)
        counts = model.comparison_counts(conn)
        ranked = sorted(
            movies, key=lambda m: scores.get(m.id, model.prior_score(m)), reverse=True
        )

        return StatusOut(
            n_movies=len(movies),
            n_comparisons_decisive=model.count_decisive(conn, source=model.ComparisonSource.USER),
            n_comparisons_total=model.count_total(conn, source=model.ComparisonSource.USER),
            n_rating_derived=model.count_total(conn, source=model.ComparisonSource.RATING),
            n_components=len(components),
            is_connected=len(components) <= 1,
            kendall_tau_vs_prev=run.kendall_tau_vs_prev if run else None,
            last_fit_timestamp=run.timestamp if run else None,
            top20=[_to_ranked_movie_out(m, scores, counts) for m in ranked[:20]],
        )
    finally:
        conn.close()


@app.get("/next_pair", response_model=PairOut)
def next_pair(
    locked_movie_id: int | None = Query(
        default=None,
        description="If set, every pair returned includes this movie, "
        "paired against whichever opponent is most uncertain.",
    ),
    locked_rating: float | None = Query(
        default=None,
        description="If set (and locked_movie_id isn't), every pair "
        "returned is drawn from movies rated this many stars.",
    ),
) -> PairOut:
    """The next pair of movies to compare, chosen by active learning."""
    if locked_movie_id is not None and locked_rating is not None:
        raise HTTPException(
            status_code=400, detail="cannot set both locked_movie_id and locked_rating"
        )
    conn = get_conn()
    try:
        if locked_movie_id is not None and model.get_movie(conn, locked_movie_id) is None:
            raise HTTPException(
                status_code=404, detail=f"movie id {locked_movie_id} not found"
            )
        selection = model.select_next_pair(
            conn, locked_id=locked_movie_id, locked_rating=locked_rating
        )
        if selection is None:
            raise HTTPException(
                status_code=400,
                detail="Need at least 2 movies. Run ingest.py first.",
            )
        (a_id, b_id), strategy = selection
        a_movie = model.get_movie(conn, a_id)
        b_movie = model.get_movie(conn, b_id)
        assert a_movie is not None and b_movie is not None

        return PairOut(
            a=_to_movie_out(a_movie, conn, fetch_missing_poster=True),
            b=_to_movie_out(b_movie, conn, fetch_missing_poster=True),
            strategy=strategy,
        )
    finally:
        conn.close()


@app.post("/compare", response_model=CompareResult)
def compare(req: CompareRequest) -> CompareResult:
    """Record a comparison result. Does not refit -- see POST /refit."""
    if req.movie_a_id == req.movie_b_id:
        raise HTTPException(status_code=400, detail="movie_a_id and movie_b_id must differ")
    if req.winner_id is not None and req.winner_id not in (req.movie_a_id, req.movie_b_id):
        raise HTTPException(status_code=400, detail="winner_id must be one of the two movies")

    conn = get_conn()
    try:
        for movie_id in (req.movie_a_id, req.movie_b_id):
            if model.get_movie(conn, movie_id) is None:
                raise HTTPException(status_code=404, detail=f"movie id {movie_id} not found")

        model.add_comparison(conn, req.movie_a_id, req.movie_b_id, req.winner_id)
        n_decisive = model.count_decisive(conn, source=model.ComparisonSource.USER)
        return CompareResult(n_comparisons_decisive=n_decisive)
    finally:
        conn.close()


@app.get("/movies", response_model=MovieListOut)
def list_movies() -> MovieListOut:
    """All movies (id/title/year only), for the history filter's datalist."""
    conn = get_conn()
    try:
        movies = sorted(
            model.get_movies(conn), key=lambda m: (m.title.lower(), m.year or 0)
        )
        return MovieListOut(
            movies=[MovieRefOut(id=m.id, title=m.title, year=m.year) for m in movies]
        )
    finally:
        conn.close()


@app.get("/comparisons", response_model=ComparisonListOut)
def list_comparisons(
    limit: int = Query(default=50, ge=1, le=200),
    before_id: int | None = Query(
        default=None, description="Return comparisons older than this id."
    ),
    movie_id: int | None = Query(
        default=None, description="Only comparisons involving this movie."
    ),
    source: model.ComparisonSource | None = Query(
        default=None, description="Only comparisons from this source."
    ),
) -> ComparisonListOut:
    """A page of comparison history, most recent first, optionally filtered
    to comparisons involving `movie_id` and/or coming from `source`."""
    conn = get_conn()
    try:
        records = model.list_comparisons(
            conn, limit=limit, before_id=before_id, movie_id=movie_id, source=source
        )
        movies_by_id = {m.id: m for m in model.get_movies(conn)}
        out = [_to_comparison_out(r, movies_by_id) for r in records]
        return ComparisonListOut(
            comparisons=[c for c in out if c is not None],
            total=model.count_total(conn, movie_id=movie_id, source=source),
        )
    finally:
        conn.close()


@app.post("/sync", response_model=SyncResult)
def sync() -> SyncResult:
    """Re-read data/diary.csv and/or data/ratings.csv and upsert movies from
    them -- the same thing `python ingest.py --data-dir data/` does, just
    without leaving the browser. Export a fresh copy from Letterboxd and
    drop it in data/ first; this only re-reads files already on disk."""
    conn = get_conn()
    try:
        try:
            n_ingested = ingest_from_data_dir()
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))
        n_total = len(model.get_movies(conn))
        return SyncResult(n_movies_ingested=n_ingested, n_movies_total=n_total)
    finally:
        conn.close()


@app.post("/seed_rating_comparisons", response_model=SeedResult)
def seed_rating_comparisons() -> SeedResult:
    """Insert a permanent comparison for every pair of differently-rated
    movies that doesn't already have one, so you can browse, edit, or delete
    them in the Comparison history panel just like any other comparison."""
    conn = get_conn()
    try:
        n_added = model.seed_rating_comparisons(conn)
        return SeedResult(n_added=n_added, n_comparisons_total=model.count_total(conn))
    finally:
        conn.close()


@app.patch("/comparisons/{comparison_id}", response_model=CompareResult)
def update_comparison(comparison_id: int, req: UpdateComparisonRequest) -> CompareResult:
    """Reassign a past comparison's winner. Does not refit -- see POST /refit."""
    conn = get_conn()
    try:
        record = model.get_comparison(conn, comparison_id)
        if record is None:
            raise HTTPException(
                status_code=404, detail=f"comparison id {comparison_id} not found"
            )
        if req.winner_id is not None and req.winner_id not in (
            record.movie_a_id,
            record.movie_b_id,
        ):
            raise HTTPException(
                status_code=400,
                detail="winner_id must be one of the two movies in this comparison",
            )
        model.update_comparison_winner(conn, comparison_id, req.winner_id)
        n_decisive = model.count_decisive(conn, source=model.ComparisonSource.USER)
        return CompareResult(n_comparisons_decisive=n_decisive)
    finally:
        conn.close()


@app.delete("/comparisons/{comparison_id}", response_model=CompareResult)
def delete_comparison(comparison_id: int) -> CompareResult:
    """Permanently remove a past comparison. Does not refit -- see POST /refit."""
    conn = get_conn()
    try:
        record = model.get_comparison(conn, comparison_id)
        if record is None:
            raise HTTPException(
                status_code=404, detail=f"comparison id {comparison_id} not found"
            )
        model.delete_comparison(conn, comparison_id)
        n_decisive = model.count_decisive(conn, source=model.ComparisonSource.USER)
        return CompareResult(n_comparisons_decisive=n_decisive)
    finally:
        conn.close()


@app.post("/refit", response_model=RefitResult)
def refit() -> RefitResult:
    """Fit the Bradley-Terry model over all current comparisons right now."""
    conn = get_conn()
    try:
        t0 = time.perf_counter()
        model.fit_scores(conn)
        elapsed = time.perf_counter() - t0
        run = model.latest_model_run(conn)
        if run is None:
            raise HTTPException(status_code=400, detail="No movies to fit. Run ingest.py first.")
        n_decisive = model.count_decisive(conn, source=model.ComparisonSource.USER)
        return RefitResult(
            n_comparisons_decisive=n_decisive,
            kendall_tau_vs_prev=run.kendall_tau_vs_prev,
            elapsed_seconds=round(elapsed, 2),
        )
    finally:
        conn.close()


@app.get("/export")
def export() -> Response:
    """The full ranking as a downloadable CSV."""
    conn = get_conn()
    try:
        movies = model.get_movies(conn)
        scores = model.current_scores(conn)
        counts = model.comparison_counts(conn)
        ranked = sorted(
            movies, key=lambda m: scores.get(m.id, model.prior_score(m)), reverse=True
        )

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            ["rank", "title", "year", "score", "letterboxd_rating", "n_comparisons"]
        )
        for rank, m in enumerate(ranked, start=1):
            writer.writerow(
                [
                    rank,
                    m.title,
                    m.year,
                    round(scores.get(m.id, 0.0), 4),
                    m.letterboxd_rating,
                    counts.get(m.id, 0),
                ]
            )

        return Response(
            content=buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=ranking.csv"},
        )
    finally:
        conn.close()


@app.get("/export_comparisons")
def export_comparisons() -> Response:
    """Your real comparisons (not rating-derived -- those regenerate
    instantly via "Seed from ratings" in the new app) as JSON, keyed by
    title/year rather than internal ids, which aren't portable between
    this app and the static one. One-time migration into the static
    version -- see the README's "Migrating to the static version"."""
    conn = get_conn()
    try:
        movies_by_id = {m.id: m for m in model.get_movies(conn)}
        rows = []
        for c in model.all_comparisons(conn, source=model.ComparisonSource.USER):
            movie_a = movies_by_id.get(c.movie_a_id)
            movie_b = movies_by_id.get(c.movie_b_id)
            if movie_a is None or movie_b is None:
                continue  # Orphaned row (shouldn't happen); skip rather than fail the export.
            winner = "a" if c.winner_id == movie_a.id else "b" if c.winner_id == movie_b.id else None
            rows.append(
                {
                    "movieA": {"title": movie_a.title, "year": movie_a.year},
                    "movieB": {"title": movie_b.title, "year": movie_b.year},
                    "winner": winner,
                    "timestamp": c.timestamp,
                }
            )

        payload = {
            "format": "letterboxd-ranker-comparisons-v1",
            "exportedAt": datetime.now(timezone.utc).isoformat(),
            "comparisons": rows,
        }
        return Response(
            content=json.dumps(payload),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=comparisons.json"},
        )
    finally:
        conn.close()
