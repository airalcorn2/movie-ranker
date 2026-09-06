"""Bradley-Terry model fitting (via choix) and active-learning pair selection."""

from __future__ import annotations

import itertools
import json
import random
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TypeAlias

import choix
import networkx as nx
import numpy as np
from scipy.stats import kendalltau

MovieId: TypeAlias = int
ScoreMap: TypeAlias = dict[MovieId, float]
Pair: TypeAlias = tuple[MovieId, MovieId]

# -- Tunables -----------------------------------------------------------

# Below this many decisive (non-skip) comparisons, prefer sampling pairs
# with adjacent star ratings over model-uncertainty sampling.
EARLY_STAGE_LIMIT = 50

# Refit the model every this many *new* decisive comparisons rather than on
# every single click, to keep the UI snappy.
REFIT_INTERVAL = 15

# choix regularization strength. Keeps scores from diverging to +/-infinity
# when the comparison graph is small, sparse, or (temporarily) disconnected.
ALPHA = 0.1

# How many synthetic "pseudo-comparisons" derived from the star-rating prior
# to mix in per adjacent-rating movie pair, before any real comparisons
# exist. Decays to 0 as real data accumulates (see `_pseudo_prior_copies`).
PSEUDO_PRIOR_MAX_COPIES = 3
PSEUDO_PRIOR_DECAY_STEP = 20  # Real comparisons per copy shed.
PSEUDO_PRIOR_MAX_PARTNERS = 5  # Cap pseudo-pairs generated per movie.

# Star ratings within this gap are considered "adjacent" for both pseudo-
# prior generation and early-stage active sampling.
ADJACENT_RATING_GAP = 0.5

# Candidate pool size for uncertainty-based sampling once a model exists,
# so we don't enumerate all O(n^2) pairs for large collections.
UNCERTAINTY_POOL_SIZE = 200

# ------------------------------------------------------------------------


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
class Comparison:
    """A stored comparison, as used internally for model fitting."""

    movie_a_id: MovieId
    movie_b_id: MovieId
    winner_id: MovieId | None  # None means skip / can't decide.

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Comparison:
        return cls(row["movie_a_id"], row["movie_b_id"], row["winner_id"])

    @property
    def loser_id(self) -> MovieId | None:
        if self.winner_id is None:
            return None
        return self.movie_b_id if self.winner_id == self.movie_a_id else self.movie_a_id


@dataclass(frozen=True, slots=True)
class ComparisonRecord:
    """A stored comparison, for the editable history view (unlike the bare
    `Comparison` used internally for model fitting, this carries its row id
    and timestamp)."""

    id: int
    movie_a_id: MovieId
    movie_b_id: MovieId
    winner_id: MovieId | None  # None means skip / can't decide.
    timestamp: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ComparisonRecord:
        return cls(
            id=row["id"],
            movie_a_id=row["movie_a_id"],
            movie_b_id=row["movie_b_id"],
            winner_id=row["winner_id"],
            timestamp=row["timestamp"],
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

    CONNECTIVITY = "connectivity"
    ADJACENT_RATING = "adjacent_rating"
    UNCERTAINTY = "uncertainty"
    RANDOM_FALLBACK = "random_fallback"


def get_movies(conn: sqlite3.Connection) -> list[Movie]:
    """All movies, ordered by id."""
    rows = conn.execute("SELECT * FROM movies ORDER BY id").fetchall()
    return [Movie.from_row(row) for row in rows]


def get_movie(conn: sqlite3.Connection, movie_id: MovieId) -> Movie | None:
    """A single movie by id, or None if it doesn't exist."""
    row = conn.execute("SELECT * FROM movies WHERE id = ?", (movie_id,)).fetchone()
    return Movie.from_row(row) if row is not None else None


def count_decisive(conn: sqlite3.Connection) -> int:
    """Count of comparisons with a recorded winner (skips excluded)."""
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM comparisons WHERE winner_id IS NOT NULL"
    ).fetchone()
    return row["c"]


def count_total(conn: sqlite3.Connection, movie_id: MovieId | None = None) -> int:
    """Total comparisons, optionally restricted to ones involving `movie_id`."""
    if movie_id is None:
        row = conn.execute("SELECT COUNT(*) AS c FROM comparisons").fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM comparisons WHERE movie_a_id = ? OR movie_b_id = ?",
            (movie_id, movie_id),
        ).fetchone()
    return row["c"]


def list_comparisons(
    conn: sqlite3.Connection,
    limit: int = 50,
    before_id: int | None = None,
    movie_id: MovieId | None = None,
) -> list[ComparisonRecord]:
    """Most recent comparisons first, optionally restricted to ones involving
    `movie_id` (e.g. to re-evaluate everything compared against a movie you
    just rewatched).

    Paginate with `before_id` (return comparisons older than that id) rather
    than OFFSET: OFFSET counts rows from the *current* start of the ordering,
    so a page fetched after new comparisons have been inserted (which sort
    to the front, being newest) silently skips/duplicates rows. An id cursor
    doesn't have that problem, since every id it excludes is unaffected by
    later inserts or by deleting a different row.
    """
    clauses = []
    params: list[int] = []
    if before_id is not None:
        clauses.append("id < ?")
        params.append(before_id)
    if movie_id is not None:
        clauses.append("(movie_a_id = ? OR movie_b_id = ?)")
        params.extend([movie_id, movie_id])
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)

    rows = conn.execute(
        f"""
        SELECT id, movie_a_id, movie_b_id, winner_id, timestamp
        FROM comparisons {where} ORDER BY id DESC LIMIT ?
        """,
        params,
    ).fetchall()
    return [ComparisonRecord.from_row(row) for row in rows]


def get_comparison(
    conn: sqlite3.Connection, comparison_id: int
) -> ComparisonRecord | None:
    row = conn.execute(
        "SELECT id, movie_a_id, movie_b_id, winner_id, timestamp FROM comparisons WHERE id = ?",
        (comparison_id,),
    ).fetchone()
    return ComparisonRecord.from_row(row) if row is not None else None


def update_comparison_winner(
    conn: sqlite3.Connection, comparison_id: int, winner_id: MovieId | None
) -> None:
    """Reassign (or clear, with None) a comparison's winner."""
    conn.execute(
        "UPDATE comparisons SET winner_id = ? WHERE id = ?", (winner_id, comparison_id)
    )
    conn.commit()


def delete_comparison(conn: sqlite3.Connection, comparison_id: int) -> None:
    """Permanently remove a comparison."""
    conn.execute("DELETE FROM comparisons WHERE id = ?", (comparison_id,))
    conn.commit()


def _all_comparisons(conn: sqlite3.Connection) -> list[Comparison]:
    """Every comparison, decisive and skipped alike."""
    rows = conn.execute(
        "SELECT movie_a_id, movie_b_id, winner_id FROM comparisons"
    ).fetchall()
    return [Comparison.from_row(row) for row in rows]


def comparison_counts(conn: sqlite3.Connection) -> dict[MovieId, int]:
    """Per-movie count of comparisons shown (decisive + skipped)."""
    counts: dict[MovieId, int] = defaultdict(int)
    for c in _all_comparisons(conn):
        counts[c.movie_a_id] += 1
        counts[c.movie_b_id] += 1
    return counts


def compared_pairs(conn: sqlite3.Connection) -> set[frozenset[MovieId]]:
    """Set of frozenset({a_id, b_id}) already shown, so we don't repeat."""
    return {frozenset((c.movie_a_id, c.movie_b_id)) for c in _all_comparisons(conn)}


def connected_components(
    conn: sqlite3.Connection, movies: list[Movie]
) -> list[set[MovieId]]:
    """Connected components over movies, edges from decisive comparisons.

    A movie with zero decisive comparisons is its own singleton component.
    """
    g = nx.Graph()
    g.add_nodes_from(m.id for m in movies)
    for c in _all_comparisons(conn):
        if c.winner_id is not None:
            g.add_edge(c.movie_a_id, c.movie_b_id)
    return list(nx.connected_components(g))


def prior_score(movie: Movie) -> float:
    """Prior BT skill score derived from the Letterboxd star rating.

    Centered at 0 for a 3-star rating (or no rating at all), +/-1 per star.
    """
    if movie.letterboxd_rating is None:
        return 0.0
    return movie.letterboxd_rating - 3.0


def _pseudo_prior_copies(n_real: int) -> int:
    """Pseudo-comparison repetition count for a given amount of real data."""
    return max(0, PSEUDO_PRIOR_MAX_COPIES - n_real // PSEUDO_PRIOR_DECAY_STEP)


def _pseudo_prior_data(
    movies: list[Movie], id_to_idx: dict[MovieId, int], n_real: int
) -> list[tuple[int, int]]:
    """Synthetic (winner_idx, loser_idx) pairs encoding the star-rating prior.

    Only generated between movies with adjacent ratings, capped per movie,
    so this stays linear in the number of rated movies. Weight (repetition
    count) decays toward 0 as real comparisons accumulate.
    """
    copies = _pseudo_prior_copies(n_real)
    if copies == 0:
        return []

    rated = [m for m in movies if m.letterboxd_rating is not None]
    rated.sort(key=lambda m: m.letterboxd_rating)  # type: ignore[arg-type,return-value]

    data: list[tuple[int, int]] = []
    for i, movie in enumerate(rated):
        partners = 0
        for other in rated[i + 1 :]:
            gap = other.letterboxd_rating - movie.letterboxd_rating  # type: ignore[operator]
            if gap > ADJACENT_RATING_GAP:
                break
            if gap == 0:
                continue  # Tie -- no directional information.
            winner_idx = id_to_idx[other.id]
            loser_idx = id_to_idx[movie.id]
            data.extend([(winner_idx, loser_idx)] * copies)
            partners += 1
            if partners >= PSEUDO_PRIOR_MAX_PARTNERS:
                break
    return data


def _real_data(
    conn: sqlite3.Connection, id_to_idx: dict[MovieId, int]
) -> list[tuple[int, int]]:
    """Real (winner_idx, loser_idx) pairs for choix, from decisive comparisons."""
    data = []
    for c in _all_comparisons(conn):
        if c.winner_id is None:
            continue
        data.append((id_to_idx[c.winner_id], id_to_idx[c.loser_id]))  # type: ignore[index]
    return data


def fit_scores(conn: sqlite3.Connection) -> ScoreMap:
    """Fit a Bradley-Terry model over all comparisons and persist the run.

    Returns {movie_id: score}. Falls back to pure prior scores if there are
    no comparisons yet at all.
    """
    movies = get_movies(conn)
    if not movies:
        return {}

    id_to_idx = {m.id: i for i, m in enumerate(movies)}
    idx_to_id = {i: m.id for i, m in enumerate(movies)}
    n = len(movies)

    real_data = _real_data(conn, id_to_idx)
    pseudo_data = _pseudo_prior_data(movies, id_to_idx, len(real_data))
    data = real_data + pseudo_data

    initial_params = np.array([prior_score(m) for m in movies])

    if not data:
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
    _store_model_run(conn, scores, n_comparisons=len(real_data))
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
    row = conn.execute(
        "SELECT * FROM model_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
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
    hasn't been through a fit yet (e.g. ingested after the last refit)."""
    run = latest_model_run(conn)
    scores: ScoreMap = dict(run.scores) if run else {}
    for m in get_movies(conn):
        scores.setdefault(m.id, prior_score(m))
    return scores


def maybe_refit(conn: sqlite3.Connection) -> ScoreMap:
    """Refit if REFIT_INTERVAL new decisive comparisons have accrued since
    the last model run."""
    n_decisive = count_decisive(conn)
    prev = latest_model_run(conn)
    prev_n = prev.n_comparisons if prev else 0
    if prev is None or n_decisive - prev_n >= REFIT_INTERVAL:
        return fit_scores(conn)
    return prev.scores


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
    exclude: set[frozenset[MovieId]] | None,
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
        if exclude is None or frozenset((a, b)) not in exclude:
            return (a, b)
    # Every pair has been shown before (tiny collection) -- allow a repeat,
    # weighted toward the least-compared movies for extra precision.
    a, b = np.random.choice(all_ids, size=2, replace=False, p=weights)
    return (int(a), int(b))


def _cross_component_pair(
    components: list[set[MovieId]],
    used_pairs: set[frozenset[MovieId]],
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
            if frozenset((a, b)) not in used_pairs:
                return (a, b)
    return None


def _rating_buckets(movies: list[Movie]) -> dict[float, list[MovieId]]:
    """Movie ids grouped by star rating, rounded to the nearest half-star."""
    buckets: dict[float, list[MovieId]] = defaultdict(list)
    for m in movies:
        key = round(m.letterboxd_rating * 2) / 2 if m.letterboxd_rating is not None else 3.0
        buckets[key].append(m.id)
    return buckets


def _adjacent_rating_pair(
    movies: list[Movie],
    used_pairs: set[frozenset[MovieId]],
    counts: dict[MovieId, int],
    max_attempts: int = 50,
) -> Pair | None:
    """A pair of movies with the same or an adjacent star rating, or None if
    none is found within `max_attempts`."""
    buckets = _rating_buckets(movies)
    keys = sorted(buckets.keys())
    if not keys:
        return None
    for _ in range(max_attempts):
        key = random.choice(keys)
        idx = keys.index(key)
        neighbor_keys = keys[max(0, idx - 1) : idx + 2]  # Self + adjacent buckets.
        candidates = list({mid for k in neighbor_keys for mid in buckets[k]})
        if len(candidates) < 2:
            continue
        pick = _weighted_choice(candidates, counts, 2)
        if not pick or len(pick) < 2:
            continue
        a, b = pick[0], pick[1]
        if frozenset((a, b)) not in used_pairs:
            return (a, b)
    return None


def _max_uncertainty_pair(
    movies: list[Movie],
    scores: ScoreMap,
    used_pairs: set[frozenset[MovieId]],
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
        if frozenset((a_id, b_id)) in used_pairs:
            continue
        prob = 1.0 / (1.0 + np.exp(-(pool_scores[i] - pool_scores[j])))
        dist = abs(prob - 0.5)
        if best_dist is None or dist < best_dist:
            best_dist, best_pair = dist, (a_id, b_id)
    return best_pair


def select_next_pair(
    conn: sqlite3.Connection,
) -> tuple[Pair, SelectionStrategy] | None:
    """Return ((movie_a_id, movie_b_id), strategy) for the next comparison,
    or None if fewer than 2 movies exist."""
    movies = get_movies(conn)
    if len(movies) < 2:
        return None

    all_ids = [m.id for m in movies]
    used_pairs = compared_pairs(conn)
    counts = comparison_counts(conn)

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
        pair = _adjacent_rating_pair(movies, used_pairs, counts)
        if pair:
            return pair, SelectionStrategy.ADJACENT_RATING

    scores = latest_scores(conn) or fit_scores(conn)
    pair = _max_uncertainty_pair(movies, scores, used_pairs, counts)
    if pair:
        return pair, SelectionStrategy.UNCERTAINTY

    pair = _weighted_random_pair(all_ids, counts, exclude=used_pairs)
    return (pair, SelectionStrategy.RANDOM_FALLBACK) if pair else None
