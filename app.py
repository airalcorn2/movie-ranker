"""FastAPI server: /, /status, /next_pair, /compare, /comparisons, /movies, /export."""

from __future__ import annotations

import csv
import io
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

import model
from db import get_conn, init_db
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
    n_comparisons_decisive: int
    n_comparisons_total: int
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
    """Whether a comparison change triggered a refit, and where things stand."""

    refit: bool
    n_comparisons_decisive: int


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
            n_comparisons_decisive=model.count_decisive(conn),
            n_comparisons_total=model.count_total(conn),
            n_components=len(components),
            is_connected=len(components) <= 1,
            kendall_tau_vs_prev=run.kendall_tau_vs_prev if run else None,
            last_fit_timestamp=run.timestamp if run else None,
            top20=[_to_ranked_movie_out(m, scores, counts) for m in ranked[:20]],
        )
    finally:
        conn.close()


@app.get("/next_pair", response_model=PairOut)
def next_pair() -> PairOut:
    """The next pair of movies to compare, chosen by active learning."""
    conn = get_conn()
    try:
        selection = model.select_next_pair(conn)
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
    """Record a comparison result, refitting the model periodically."""
    if req.movie_a_id == req.movie_b_id:
        raise HTTPException(status_code=400, detail="movie_a_id and movie_b_id must differ")
    if req.winner_id is not None and req.winner_id not in (req.movie_a_id, req.movie_b_id):
        raise HTTPException(status_code=400, detail="winner_id must be one of the two movies")

    conn = get_conn()
    try:
        for movie_id in (req.movie_a_id, req.movie_b_id):
            if model.get_movie(conn, movie_id) is None:
                raise HTTPException(status_code=404, detail=f"movie id {movie_id} not found")

        conn.execute(
            """
            INSERT INTO comparisons (movie_a_id, movie_b_id, winner_id, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (
                req.movie_a_id,
                req.movie_b_id,
                req.winner_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

        prev_run = model.latest_model_run(conn)
        prev_ts = prev_run.timestamp if prev_run else None
        model.maybe_refit(conn)
        new_run = model.latest_model_run(conn)
        refit = new_run is not None and new_run.timestamp != prev_ts

        return CompareResult(
            refit=refit, n_comparisons_decisive=model.count_decisive(conn)
        )
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
) -> ComparisonListOut:
    """A page of comparison history, most recent first, optionally filtered
    to comparisons involving `movie_id`."""
    conn = get_conn()
    try:
        records = model.list_comparisons(
            conn, limit=limit, before_id=before_id, movie_id=movie_id
        )
        movies_by_id = {m.id: m for m in model.get_movies(conn)}
        out = [_to_comparison_out(r, movies_by_id) for r in records]
        return ComparisonListOut(
            comparisons=[c for c in out if c is not None],
            total=model.count_total(conn, movie_id=movie_id),
        )
    finally:
        conn.close()


@app.patch("/comparisons/{comparison_id}", response_model=CompareResult)
def update_comparison(comparison_id: int, req: UpdateComparisonRequest) -> CompareResult:
    """Reassign a past comparison's winner and refit immediately."""
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
        # Edits are rare, deliberate actions (unlike the high-frequency
        # /compare flow) -- refit immediately rather than waiting for
        # REFIT_INTERVAL, so the ranking reflects the correction right away.
        model.fit_scores(conn)
        return CompareResult(refit=True, n_comparisons_decisive=model.count_decisive(conn))
    finally:
        conn.close()


@app.delete("/comparisons/{comparison_id}", response_model=CompareResult)
def delete_comparison(comparison_id: int) -> CompareResult:
    """Permanently remove a past comparison and refit immediately."""
    conn = get_conn()
    try:
        record = model.get_comparison(conn, comparison_id)
        if record is None:
            raise HTTPException(
                status_code=404, detail=f"comparison id {comparison_id} not found"
            )
        model.delete_comparison(conn, comparison_id)
        model.fit_scores(conn)
        return CompareResult(refit=True, n_comparisons_decisive=model.count_decisive(conn))
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
