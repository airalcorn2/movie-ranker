"""Bradley-Terry model fitting (via choix) and active-learning pair selection."""

from __future__ import annotations

import itertools
import json
import random
import sqlite3
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import KeysView, TypeAlias

import choix
import networkx as nx
import numpy as np
from scipy.stats import kendalltau

MovieId: TypeAlias = int
ScoreMap: TypeAlias = dict[MovieId, float]
Pair: TypeAlias = tuple[MovieId, MovieId]
PairKey: TypeAlias = tuple[MovieId, MovieId]  # Always (min(a, b), max(a, b)).

# -- Tunables -----------------------------------------------------------

# Below this many decisive (non-skip) comparisons, prefer sampling pairs
# with the same star rating over model-uncertainty sampling.
EARLY_STAGE_LIMIT = 50

# choix regularization strength. Keeps scores from diverging to +/-infinity
# when the comparison graph is small, sparse, or (temporarily) disconnected.
ALPHA = 0.1

# Candidate pool size for uncertainty-based sampling once a model exists,
# so we don't enumerate all O(n^2) pairs for large collections.
UNCERTAINTY_POOL_SIZE = 200

# ------------------------------------------------------------------------


class ComparisonSource(str, Enum):
    """Where a comparison came from."""

    USER = "user"  # A real click in the comparison UI.
    RATING = "rating"  # Auto-generated from a Letterboxd star-rating difference (see seed_rating_comparisons).


@dataclass(frozen=True, slots=True)
class Movie:
    """A movie, as stored in the `movies` table."""

    id: MovieId
    title: str
    year: int | None
    letterboxd_rating: float | None
    watched_date: str | None
    poster_url: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Movie:
        return cls(
            id=row["id"],
            title=row["title"],
            year=row["year"],
            letterboxd_rating=row["letterboxd_rating"],
            watched_date=row["watched_date"],
            poster_url=row["poster_url"],
        )


@dataclass(frozen=True, slots=True)
class ComparisonRecord:
    """A stored comparison, for the editable history view."""

    id: int
    movie_a_id: MovieId
    movie_b_id: MovieId
    winner_id: MovieId | None  # None means skip / can't decide.
    timestamp: str
    source: ComparisonSource

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ComparisonRecord:
        return cls(
            id=row["id"],
            movie_a_id=row["movie_a_id"],
            movie_b_id=row["movie_b_id"],
            winner_id=row["winner_id"],
            timestamp=row["timestamp"],
            source=ComparisonSource(row["source"]),
        )


@dataclass(frozen=True, slots=True)
class ModelRun:
    """A persisted Bradley-Terry fit: the resulting scores plus fit metadata."""

    timestamp: str
    n_comparisons: int
    scores: ScoreMap
    kendall_tau_vs_prev: float | None


class SelectionStrategy(str, Enum):
    """Which rule in `select_next_pair`'s priority order produced a pair."""

    LOCKED = "locked"
    CONNECTIVITY = "connectivity"
    SAME_RATING = "same_rating"
    UNCERTAINTY = "uncertainty"
    RANDOM_FALLBACK = "random_fallback"


def _pair_key(a: MovieId, b: MovieId) -> PairKey:
    return (a, b) if a < b else (b, a)


def get_movies(conn: sqlite3.Connection) -> list[Movie]:
    """All movies, ordered by id."""
    rows = conn.execute("SELECT * FROM movies ORDER BY id").fetchall()
    return [Movie.from_row(row) for row in rows]


def get_movie(conn: sqlite3.Connection, movie_id: MovieId) -> Movie | None:
    """A single movie by id, or None if it doesn't exist."""
    row = conn.execute("SELECT * FROM movies WHERE id = ?", (movie_id,)).fetchone()
    return Movie.from_row(row) if row is not None else None


def count_decisive(
    conn: sqlite3.Connection, source: ComparisonSource | None = None
) -> int:
    """Count of comparisons with a recorded winner (skips excluded),
    optionally restricted to one source."""
    if source is None:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM comparisons WHERE winner_id IS NOT NULL"
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM comparisons WHERE winner_id IS NOT NULL AND source = ?",
            (source.value,),
        ).fetchone()
    return row["c"]


def count_total(
    conn: sqlite3.Connection,
    movie_id: MovieId | None = None,
    source: ComparisonSource | None = None,
) -> int:
    """Total comparisons, optionally restricted to ones involving `movie_id`
    and/or coming from `source`."""
    clauses = []
    params: list = []
    if movie_id is not None:
        clauses.append("(movie_a_id = ? OR movie_b_id = ?)")
        params.extend([movie_id, movie_id])
    if source is not None:
        clauses.append("source = ?")
        params.append(source.value)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    row = conn.execute(
        f"SELECT COUNT(*) AS c FROM comparisons {where}", params
    ).fetchone()
    return row["c"]


def list_comparisons(
    conn: sqlite3.Connection,
    limit: int = 50,
    before_id: int | None = None,
    movie_id: MovieId | None = None,
    source: ComparisonSource | None = None,
) -> list[ComparisonRecord]:
    """Most recent comparisons first, optionally restricted to ones involving
    `movie_id` (e.g., to re-evaluate everything compared against a movie you
    just rewatched) and/or coming from `source`.

    Paginate with `before_id` (return comparisons older than that id) rather
    than OFFSET: OFFSET counts rows from the *current* start of the ordering,
    so a page fetched after new comparisons have been inserted (which sort
    to the front, being newest) silently skips/duplicates rows. An id cursor
    doesn't have that problem, since every id it excludes is unaffected by
    later inserts or by deleting a different row.
    """
    clauses = []
    params: list = []
    if before_id is not None:
        clauses.append("id < ?")
        params.append(before_id)
    if movie_id is not None:
        clauses.append("(movie_a_id = ? OR movie_b_id = ?)")
        params.extend([movie_id, movie_id])
    if source is not None:
        clauses.append("source = ?")
        params.append(source.value)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)

    rows = conn.execute(
        f"""
        SELECT id, movie_a_id, movie_b_id, winner_id, timestamp, source
        FROM comparisons {where} ORDER BY id DESC LIMIT ?
        """,
        params,
    ).fetchall()
    return [ComparisonRecord.from_row(row) for row in rows]


def get_comparison(
    conn: sqlite3.Connection, comparison_id: int
) -> ComparisonRecord | None:
    """A single comparison by id, or None if it doesn't exist."""
    row = conn.execute(
        "SELECT id, movie_a_id, movie_b_id, winner_id, timestamp, source FROM comparisons WHERE id = ?",
        (comparison_id,),
    ).fetchone()
    return ComparisonRecord.from_row(row) if row is not None else None


# -- In-memory comparison caches -----------------------------------------
#
# compared_pairs()/comparison_counts()/connected_components() are called on
# every /next_pair and /status request. Once the comparisons table holds a
# large permanent rating-derived prior (see seed_rating_comparisons),
# re-scanning it from Python on every request measurably slows every click
# (benchmarked ~0.9s/request at ~134K rows). So each is built from the
# database once per process (or after an explicit invalidation) and kept in
# sync incrementally by add_comparison()/update_comparison_winner()/
# delete_comparison() below, rather than rebuilt on every read.

_cache_lock = threading.Lock()
_pair_counts: dict[PairKey, int] | None = (
    None  # All comparisons per pair (decisive + skipped).
)
_decisive_pair_counts: dict[PairKey, int] | None = (
    None  # Just the decisive ones (drives _graph's edges).
)
_movie_counts: dict[MovieId, int] | None = None  # All comparisons touching each movie.
_graph: nx.Graph | None = (
    None  # Nodes = every movie; edges = pairs with a decisive comparison.
)


def _invalidate_cache() -> None:
    """Force the next access to rebuild the in-memory comparison caches from
    scratch. Use after a bulk change (e.g., seed_rating_comparisons) where
    rebuilding once is cheaper than updating incrementally row by row."""
    global _pair_counts, _decisive_pair_counts, _movie_counts, _graph
    with _cache_lock:
        _pair_counts = None
        _decisive_pair_counts = None
        _movie_counts = None
        _graph = None


def _ensure_cache(conn: sqlite3.Connection) -> None:
    """Build the in-memory comparison caches if they haven't been already."""
    global _pair_counts, _decisive_pair_counts, _movie_counts, _graph
    if _pair_counts is not None:
        return
    with _cache_lock:
        if (
            _pair_counts is not None
        ):  # Another thread may have built it while we waited.
            return
        pair_counts: dict[PairKey, int] = defaultdict(int)
        decisive_pair_counts: dict[PairKey, int] = defaultdict(int)
        movie_counts: dict[MovieId, int] = defaultdict(int)
        graph = nx.Graph()
        graph.add_nodes_from(row["id"] for row in conn.execute("SELECT id FROM movies"))
        for row in conn.execute(
            "SELECT movie_a_id, movie_b_id, winner_id FROM comparisons"
        ):
            a, b, w = row["movie_a_id"], row["movie_b_id"], row["winner_id"]
            key = _pair_key(a, b)
            pair_counts[key] += 1
            movie_counts[a] += 1
            movie_counts[b] += 1
            if w is not None:
                decisive_pair_counts[key] += 1
                graph.add_edge(a, b)
        _pair_counts = dict(pair_counts)
        _decisive_pair_counts = dict(decisive_pair_counts)
        _movie_counts = dict(movie_counts)
        _graph = graph


def _adjust_decisive_edge(a: MovieId, b: MovieId, delta: int) -> None:
    """Apply `delta` to the decisive-comparison count for (a, b) and keep
    the cached graph's edge presence in sync. Caller holds `_cache_lock`."""
    assert _decisive_pair_counts is not None and _graph is not None
    key = _pair_key(a, b)
    new_count = _decisive_pair_counts.get(key, 0) + delta
    if new_count <= 0:
        _decisive_pair_counts.pop(key, None)
        if _graph.has_edge(a, b):
            _graph.remove_edge(a, b)
    else:
        _decisive_pair_counts[key] = new_count
        _graph.add_edge(a, b)


def _cache_record(a: MovieId, b: MovieId, decisive: bool, delta: int) -> None:
    """Apply `delta` to the pair/movie touch counts for (a, b), and to the
    decisive-comparison graph if `decisive`. Assumes the cache is built."""
    assert _pair_counts is not None and _movie_counts is not None
    with _cache_lock:
        key = _pair_key(a, b)
        new_pair_count = _pair_counts.get(key, 0) + delta
        if new_pair_count <= 0:
            _pair_counts.pop(key, None)
        else:
            _pair_counts[key] = new_pair_count
        _movie_counts[a] = max(0, _movie_counts.get(a, 0) + delta)
        _movie_counts[b] = max(0, _movie_counts.get(b, 0) + delta)
        if decisive:
            _adjust_decisive_edge(a, b, delta)


def comparison_counts(conn: sqlite3.Connection) -> dict[MovieId, int]:
    """Per-movie count of comparisons shown (decisive + skipped)."""
    _ensure_cache(conn)
    assert _movie_counts is not None
    return _movie_counts


def compared_pairs(conn: sqlite3.Connection) -> KeysView[PairKey]:
    """Canonical (a_id, b_id) pairs already shown, so we don't repeat one."""
    _ensure_cache(conn)
    assert _pair_counts is not None
    return _pair_counts.keys()


def connected_components(
    conn: sqlite3.Connection, movies: list[Movie]
) -> list[set[MovieId]]:
    """Connected components over movies, edges from decisive comparisons.

    A movie with zero decisive comparisons is its own singleton component.
    """
    _ensure_cache(conn)
    assert _graph is not None
    # Cheap no-op for movies already present; picks up any ingested after
    # the cache was built.
    _graph.add_nodes_from(m.id for m in movies)
    return list(nx.connected_components(_graph))


def add_comparison(
    conn: sqlite3.Connection,
    movie_a_id: MovieId,
    movie_b_id: MovieId,
    winner_id: MovieId | None,
    source: ComparisonSource = ComparisonSource.USER,
) -> None:
    """Record a comparison, keeping the in-memory caches in sync."""
    _ensure_cache(conn)
    conn.execute(
        """
        INSERT INTO comparisons (movie_a_id, movie_b_id, winner_id, timestamp, source)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            movie_a_id,
            movie_b_id,
            winner_id,
            datetime.now(timezone.utc).isoformat(),
            source.value,
        ),
    )
    conn.commit()
    _cache_record(movie_a_id, movie_b_id, decisive=winner_id is not None, delta=1)


def update_comparison_winner(
    conn: sqlite3.Connection, comparison_id: int, winner_id: MovieId | None
) -> None:
    """Reassign (or clear, with None) a comparison's winner."""
    _ensure_cache(conn)
    record = get_comparison(conn, comparison_id)
    conn.execute(
        "UPDATE comparisons SET winner_id = ? WHERE id = ?", (winner_id, comparison_id)
    )
    conn.commit()
    if record is None:
        return
    was_decisive = record.winner_id is not None
    is_decisive = winner_id is not None
    if was_decisive != is_decisive:
        # Pair/movie touch counts are unaffected by *who* won -- only
        # whether the graph has an edge for this pair changes.
        with _cache_lock:
            _adjust_decisive_edge(
                record.movie_a_id, record.movie_b_id, 1 if is_decisive else -1
            )


def delete_comparison(conn: sqlite3.Connection, comparison_id: int) -> None:
    """Permanently remove a comparison, keeping the in-memory caches in sync."""
    _ensure_cache(conn)
    record = get_comparison(conn, comparison_id)
    conn.execute("DELETE FROM comparisons WHERE id = ?", (comparison_id,))
    conn.commit()
    if record is not None:
        _cache_record(
            record.movie_a_id,
            record.movie_b_id,
            decisive=record.winner_id is not None,
            delta=-1,
        )


def prior_score(movie: Movie) -> float:
    """Prior BT skill score derived from the Letterboxd star rating.

    Centered at 0 for a 3-star rating (or no rating at all), +/-1 per star.
    """
    if movie.letterboxd_rating is None:
        return 0.0
    return movie.letterboxd_rating - 3.0


def seed_rating_comparisons(conn: sqlite3.Connection) -> int:
    """Insert one permanent comparison (source=RATING) for every pair of
    differently-rated movies that doesn't already have any comparison
    recorded at all -- whether that existing one came from you or from a
    previous seeding run -- so this is safe to re-run any time (e.g., after
    ingesting new/re-rated movies) without duplicating pairs. Ties (same
    rating) never get one, so within-tier order is always left for your own
    comparisons to settle.

    Returns the number of comparisons inserted.
    """
    _ensure_cache(conn)
    assert _pair_counts is not None

    movies = get_movies(conn)
    rated = [m for m in movies if m.letterboxd_rating is not None]
    rated.sort(key=lambda m: m.letterboxd_rating)  # type: ignore[arg-type,return-value]

    timestamp = datetime.now(timezone.utc).isoformat()
    rows_to_insert = []
    for i, movie in enumerate(rated):
        for other in rated[i + 1 :]:
            if other.letterboxd_rating == movie.letterboxd_rating:
                continue  # Tie -- no directional information.
            if _pair_key(movie.id, other.id) in _pair_counts:
                continue  # Already compared, by you or a previous seeding run.
            winner, loser = (
                (other, movie)
                if other.letterboxd_rating > movie.letterboxd_rating  # type: ignore[operator]
                else (movie, other)
            )
            rows_to_insert.append(
                (
                    winner.id,
                    loser.id,
                    winner.id,
                    timestamp,
                    ComparisonSource.RATING.value,
                )
            )

    if rows_to_insert:
        conn.executemany(
            """
            INSERT INTO comparisons (movie_a_id, movie_b_id, winner_id, timestamp, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )
        conn.commit()
        _invalidate_cache()  # Bulk change -- cheaper to rebuild once than update row by row.
    return len(rows_to_insert)


def _fit_data(
    conn: sqlite3.Connection, id_to_idx: dict[MovieId, int]
) -> list[tuple[int, int]]:
    """(winner_idx, loser_idx) pairs for choix, from every decisive
    comparison regardless of source -- a real click and a rating-derived
    comparison carry equal weight in the fit."""
    data = []
    for row in conn.execute(
        "SELECT movie_a_id, movie_b_id, winner_id FROM comparisons WHERE winner_id IS NOT NULL"
    ):
        loser_id = (
            row["movie_b_id"]
            if row["winner_id"] == row["movie_a_id"]
            else row["movie_a_id"]
        )
        data.append((id_to_idx[row["winner_id"]], id_to_idx[loser_id]))
    return data


def fit_scores(conn: sqlite3.Connection) -> ScoreMap:
    """Fit a Bradley-Terry model over all comparisons and persist the run.

    Returns {movie_id: score}. Falls back to pure prior scores if there are
    no comparisons at all yet.
    """
    movies = get_movies(conn)
    if not movies:
        return {}

    id_to_idx = {m.id: i for i, m in enumerate(movies)}
    idx_to_id = {i: m.id for i, m in enumerate(movies)}
    n = len(movies)

    data = _fit_data(conn, id_to_idx)
    initial_params = np.array([prior_score(m) for m in movies])

    if not data:
        # With nothing to fit at all, `initial_params` (the star ratings)
        # *is* the ranking.
        scores_by_idx = initial_params
    else:
        try:
            scores_by_idx = choix.ilsr_pairwise(
                n, data, alpha=ALPHA, initial_params=initial_params
            )
        except Exception:
            # ILSR can fail to converge on pathological/tiny data; the
            # Gaussian-prior MAP optimizer is a more robust fallback.
            scores_by_idx = choix.opt_pairwise(
                n, data, alpha=ALPHA, initial_params=initial_params
            )

    scores: ScoreMap = {idx_to_id[i]: float(s) for i, s in enumerate(scores_by_idx)}
    _store_model_run(conn, scores, n_comparisons=len(data))
    return scores


def _store_model_run(
    conn: sqlite3.Connection, scores: ScoreMap, n_comparisons: int
) -> None:
    """Persist a fitted score map as a new model run, recording its
    stability (Kendall's tau) against the previous run."""
    prev = latest_model_run(conn)
    tau: float | None = None
    if prev is not None:
        shared = [mid for mid in scores if mid in prev.scores]
        if len(shared) >= 2:
            a = [prev.scores[mid] for mid in shared]
            b = [scores[mid] for mid in shared]
            tau = float(kendalltau(a, b).statistic)  # type: ignore[attr-defined]

    conn.execute(
        """
        INSERT INTO model_runs (timestamp, n_comparisons, scores_json, kendall_tau_vs_prev)
        VALUES (?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            n_comparisons,
            json.dumps(scores),
            tau,
        ),
    )
    conn.commit()


def latest_model_run(conn: sqlite3.Connection) -> ModelRun | None:
    """The most recent model run, or None if the model has never been fit."""
    row = conn.execute("SELECT * FROM model_runs ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        return None
    return ModelRun(
        timestamp=row["timestamp"],
        n_comparisons=row["n_comparisons"],
        scores={int(k): v for k, v in json.loads(row["scores_json"]).items()},
        kendall_tau_vs_prev=row["kendall_tau_vs_prev"],
    )


def latest_scores(conn: sqlite3.Connection) -> ScoreMap | None:
    """Scores from the most recent model run, or None if it's never been fit."""
    run = latest_model_run(conn)
    return run.scores if run else None


def current_scores(conn: sqlite3.Connection) -> ScoreMap:
    """Latest fitted scores, filled in with prior scores for any movie that
    hasn't been through a fit yet (e.g., ingested after the last refit)."""
    run = latest_model_run(conn)
    scores: ScoreMap = dict(run.scores) if run else {}
    for m in get_movies(conn):
        scores.setdefault(m.id, prior_score(m))
    return scores


# -- Active-learning pair selection --------------------------------------


def _weighted_choice(
    ids: list[MovieId],
    counts: dict[MovieId, int],
    k: int,
    exclude_id: MovieId | None = None,
) -> list[MovieId] | None:
    """k distinct movie ids sampled without replacement, weighted toward
    movies with fewer comparisons so far."""
    pool = [i for i in ids if i != exclude_id]
    if not pool:
        return None
    weights = np.array([1.0 / (1 + counts.get(i, 0)) for i in pool])
    weights /= weights.sum()
    chosen = np.random.choice(pool, size=min(k, len(pool)), replace=False, p=weights)
    return [int(x) for x in chosen]


def _weighted_random_pair(
    all_ids: list[MovieId],
    counts: dict[MovieId, int],
    exclude: KeysView[PairKey] | None,
    max_attempts: int = 200,
) -> Pair | None:
    """A random pair of movie ids, weighted toward under-compared movies."""
    if len(all_ids) < 2:
        return None
    weights = np.array([1.0 / (1 + counts.get(i, 0)) for i in all_ids])
    weights /= weights.sum()
    for _ in range(max_attempts):
        a, b = np.random.choice(all_ids, size=2, replace=False, p=weights)
        a, b = int(a), int(b)
        if exclude is None or _pair_key(a, b) not in exclude:
            return (a, b)
    # Every pair has been shown before (tiny collection) -- allow a repeat,
    # weighted toward the least-compared movies for extra precision.
    a, b = np.random.choice(all_ids, size=2, replace=False, p=weights)
    return (int(a), int(b))


def _cross_component_pair(
    components: list[set[MovieId]],
    used_pairs: KeysView[PairKey],
    counts: dict[MovieId, int],
    max_attempts: int = 20,
) -> Pair | None:
    """Bridge two components, smallest first, avoiding already-used pairs."""
    ordered = sorted(components, key=len)
    for comp_a, comp_b in itertools.combinations(ordered, 2):
        list_a, list_b = list(comp_a), list(comp_b)
        for _ in range(max_attempts):
            pick_a = _weighted_choice(list_a, counts, 1)
            pick_b = _weighted_choice(list_b, counts, 1)
            if not pick_a or not pick_b:
                break
            a, b = pick_a[0], pick_b[0]
            if _pair_key(a, b) not in used_pairs:
                return (a, b)
    return None


def _rating_buckets(movies: list[Movie]) -> dict[float, list[MovieId]]:
    """Movie ids grouped by star rating, rounded to the nearest half-star."""
    buckets: dict[float, list[MovieId]] = defaultdict(list)
    for m in movies:
        key = (
            round(m.letterboxd_rating * 2) / 2
            if m.letterboxd_rating is not None
            else 3.0
        )
        buckets[key].append(m.id)
    return buckets


def _same_rating_pair(
    movies: list[Movie],
    used_pairs: KeysView[PairKey],
    counts: dict[MovieId, int],
    max_attempts: int = 50,
) -> Pair | None:
    """A pair of movies with the same star rating, or None if none is found
    within `max_attempts`.

    Restricted to *same* rating, not merely adjacent: the star rating
    generally predicts the winner of a cross-tier pair, so those aren't
    genuinely uncertain -- same-tier pairs are where your judgment actually
    adds information the rating alone can't.
    """
    buckets = _rating_buckets(movies)
    keys = list(buckets.keys())
    if not keys:
        return None
    for _ in range(max_attempts):
        candidates = buckets[random.choice(keys)]
        if len(candidates) < 2:
            continue
        pick = _weighted_choice(candidates, counts, 2)
        if not pick or len(pick) < 2:
            continue
        a, b = pick[0], pick[1]
        if _pair_key(a, b) not in used_pairs:
            return (a, b)
    return None


def _max_uncertainty_pair(
    movies: list[Movie],
    scores: ScoreMap,
    used_pairs: KeysView[PairKey],
    counts: dict[MovieId, int],
) -> Pair | None:
    """The unused pair, from a sampled pool weighted toward under-compared
    movies, whose predicted win probability is closest to 50/50 (maximum
    uncertainty). None if no unused pair is found in the pool."""
    ids = [m.id for m in movies if m.id in scores]
    if len(ids) < 2:
        return None
    pool = _weighted_choice(ids, counts, min(len(ids), UNCERTAINTY_POOL_SIZE))
    if not pool or len(pool) < 2:
        return None

    pool_scores = np.array([scores[i] for i in pool])
    best_pair: Pair | None = None
    best_dist: float | None = None
    for i, j in itertools.combinations(range(len(pool)), 2):
        a_id, b_id = pool[i], pool[j]
        if _pair_key(a_id, b_id) in used_pairs:
            continue
        prob = 1.0 / (1.0 + np.exp(-(pool_scores[i] - pool_scores[j])))
        dist = abs(prob - 0.5)
        if best_dist is None or dist < best_dist:
            best_dist, best_pair = dist, (a_id, b_id)
    return best_pair


def _locked_pair(
    movies: list[Movie],
    scores: ScoreMap,
    locked_id: MovieId,
    used_pairs: KeysView[PairKey] | None,
    counts: dict[MovieId, int],
) -> Pair | None:
    """The pair (locked_id, opponent) whose predicted win probability is
    closest to 50/50, so the locked movie's own score converges fast.

    `used_pairs`: pass the real cache to avoid repeats, or None to allow
    them once every opponent has already been compared against locked_id.
    """
    if locked_id not in scores:
        return None
    candidates = [m.id for m in movies if m.id != locked_id]
    if used_pairs is not None:
        candidates = [
            m_id for m_id in candidates if _pair_key(locked_id, m_id) not in used_pairs
        ]
    if not candidates:
        return None

    pool = _weighted_choice(candidates, counts, min(len(candidates), UNCERTAINTY_POOL_SIZE))
    if not pool:
        return None

    locked_score = scores[locked_id]
    best_pair: Pair | None = None
    best_dist: float | None = None
    for opp_id in pool:
        prob = 1.0 / (1.0 + np.exp(-(locked_score - scores.get(opp_id, 0.0))))
        dist = abs(prob - 0.5)
        if best_dist is None or dist < best_dist:
            best_dist, best_pair = dist, (locked_id, opp_id)
    return best_pair


def select_next_pair(
    conn: sqlite3.Connection, locked_id: MovieId | None = None
) -> tuple[Pair, SelectionStrategy] | None:
    """Return ((movie_a_id, movie_b_id), strategy) for the next comparison,
    or None if fewer than 2 movies exist.

    `locked_id`: if given (and it exists), every pair returned includes it,
    paired against whichever opponent is most uncertain -- this overrides
    the usual connectivity/same-rating/uncertainty priority order, since the
    point of locking a movie is to focus entirely on nailing down its score.
    """
    movies = get_movies(conn)
    if len(movies) < 2:
        return None

    all_ids = [m.id for m in movies]
    used_pairs = compared_pairs(conn)
    counts = comparison_counts(conn)

    if locked_id is not None and locked_id in all_ids:
        scores = current_scores(conn)
        pair = _locked_pair(movies, scores, locked_id, used_pairs, counts)
        if not pair:
            # Already compared against every other movie -- allow a repeat
            # rather than falling back to the unrelated global strategy.
            pair = _locked_pair(movies, scores, locked_id, None, counts)
        if pair:
            return pair, SelectionStrategy.LOCKED

    total_possible = len(all_ids) * (len(all_ids) - 1) // 2
    if len(used_pairs) >= total_possible:
        pair = _weighted_random_pair(all_ids, counts, exclude=None)
        return (pair, SelectionStrategy.RANDOM_FALLBACK) if pair else None

    # Keep the comparison graph connected -- top priority.
    components = connected_components(conn, movies)
    if len(components) > 1:
        pair = _cross_component_pair(components, used_pairs, counts)
        if pair:
            return pair, SelectionStrategy.CONNECTIVITY

    n_decisive = count_decisive(conn)
    if n_decisive < EARLY_STAGE_LIMIT:
        pair = _same_rating_pair(movies, used_pairs, counts)
        if pair:
            return pair, SelectionStrategy.SAME_RATING

    # current_scores(), not fit_scores(): fitting is manual-only now (see
    # POST /refit), so if nothing has ever been fit yet we fall back to
    # prior-only scores here rather than silently triggering a fit on what
    # is otherwise just a GET request.
    scores = current_scores(conn)
    pair = _max_uncertainty_pair(movies, scores, used_pairs, counts)
    if pair:
        return pair, SelectionStrategy.UNCERTAINTY

    pair = _weighted_random_pair(all_ids, counts, exclude=used_pairs)
    return (pair, SelectionStrategy.RANDOM_FALLBACK) if pair else None
