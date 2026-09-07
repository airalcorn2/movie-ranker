/**
 * UI controller for the static (GitHub Pages / no-server) version.
 *
 * Holds the whole session's movies/comparisons/model-runs in memory (a
 * browser tab can easily afford that, even at 100K+ rows), persists every
 * mutation through storage.js, and rebuilds model.js's caches from the
 * in-memory comparisons array after each one -- see model.js's top
 * comment for why that's simpler and safer here than incremental updates.
 */

import * as storage from "./storage.js";
import * as M from "./model.js";
import { ingestText } from "./csv.js";

const state = {
  movies: [],
  moviesById: new Map(),
  comparisons: [],
  modelRuns: [],
  meta: new Map(),
  caches: null,
  current: null, // {a, b} movie objects currently shown.
  lockedMovieId: null,
  lockedRating: null,
};

const movieLabelToId = new Map();

function $(id) {
  return document.getElementById(id);
}

function rebuildCaches() {
  state.caches = M.buildCaches(state.movies, state.comparisons);
}

function latestModelRun() {
  return state.modelRuns.length ? state.modelRuns[state.modelRuns.length - 1] : null;
}

function countDecisive(source = null) {
  return state.comparisons.filter((c) => c.winnerId != null && (source == null || c.source === source)).length;
}

function countTotal(source = null) {
  return state.comparisons.filter((c) => source == null || c.source === source).length;
}

// -- Comparison UI -----------------------------------------------------

function fmtRating(r) {
  return r == null ? "" : "★".repeat(Math.round(r)) + ` (${r})`;
}

function renderCard(prefix, movie) {
  const isLocked = movie.id === state.lockedMovieId || movie.rating === state.lockedRating;
  $(`card-${prefix}`).classList.toggle("locked-card", isLocked);
  $(`title-${prefix}`).textContent = (isLocked ? "\u{1F512} " : "") + movie.title;
  $(`year-${prefix}`).textContent = movie.year ?? "";
  $(`rating-${prefix}`).textContent = fmtRating(movie.rating);
  const ph = $(`ph-${prefix}`);
  const existingImg = document.querySelector(`#card-${prefix} img`);
  if (existingImg) existingImg.remove();
  if (movie.posterUrl) {
    ph.style.display = "none";
    const img = document.createElement("img");
    img.src = movie.posterUrl;
    img.alt = movie.title;
    ph.insertAdjacentElement("beforebegin", img);
  } else {
    ph.style.display = "flex";
    ph.textContent = movie.title;
  }
}

/** Look up posters for a and b via TMDB, if a key is saved -- optional,
 * best-effort, and re-renders the card in place once a poster arrives. */
async function maybeFetchPosters(a, b) {
  const key = state.meta.get("tmdbApiKey");
  if (!key) return;
  for (const movie of [a, b]) {
    if (movie.posterUrl) continue;
    try {
      const params = new URLSearchParams({ api_key: key, query: movie.title });
      if (movie.year) params.set("year", String(movie.year));
      const res = await fetch(`https://api.themoviedb.org/3/search/movie?${params}`);
      if (!res.ok) continue;
      const data = await res.json();
      const hit = data.results && data.results[0];
      if (!hit || !hit.poster_path) continue;
      movie.posterUrl = `https://image.tmdb.org/t/p/w342${hit.poster_path}`;
      await storage.putMovies([movie]);
      if (state.current && (state.current.a.id === movie.id || state.current.b.id === movie.id)) {
        renderCard(state.current.a.id === movie.id ? "a" : "b", movie);
      }
    } catch (e) {
      // Posters are optional; ignore failures.
    }
  }
}

function loadNextPair() {
  $("loading").style.display = "none";
  $("vs-row").style.display = "none";
  $("empty-state").style.display = "none";
  $("banner").style.display = "none";

  if (state.movies.length < 2) {
    $("empty-state").style.display = "block";
    return;
  }
  $("loading").style.display = "block";

  const scores = M.currentScores(state.movies, latestModelRun());
  const sel = M.selectNextPair(state.movies, state.caches, scores, {
    lockedId: state.lockedMovieId,
    lockedRating: state.lockedRating,
  });
  $("loading").style.display = "none";
  if (!sel) {
    $("banner").textContent = "Could not find a pair to compare.";
    $("banner").style.display = "block";
    return;
  }
  const a = state.moviesById.get(sel.pair[0]);
  const b = state.moviesById.get(sel.pair[1]);
  state.current = { a, b };
  renderCard("a", a);
  renderCard("b", b);
  $("vs-row").style.display = "grid";
  maybeFetchPosters(a, b);
}

async function submitChoice(winnerId) {
  if (!state.current) return;
  const { a, b } = state.current;
  state.current = null;
  const comparison = { movieAId: a.id, movieBId: b.id, winnerId, timestamp: new Date().toISOString(), source: "user" };
  comparison.id = await storage.addComparison(comparison);
  state.comparisons.push(comparison);
  rebuildCaches();
  loadStatus();
  loadNextPair();
  refreshHistoryIfOpen();
}

function loadStatus() {
  const components = M.connectedComponents(state.movies.map((m) => m.id), state.caches.adjacency);
  const isConnected = components.length <= 1;
  const run = latestModelRun();
  const scores = M.currentScores(state.movies, run);
  const counts = state.caches.movieCounts;
  const ranked = [...state.movies].sort((x, y) => (scores.get(y.id) ?? 0) - (scores.get(x.id) ?? 0));

  const nDecisive = countDecisive("user");
  const nRatingDerived = countTotal("rating");
  const connNote = isConnected ? "" : ` &mdash; ${components.length} disconnected groups`;
  const tauNote = run && run.kendallTauVsPrev != null ? ` &mdash; stability &tau;=${run.kendallTauVsPrev.toFixed(2)}` : "";
  const ratingNote = nRatingDerived > 0 ? ` &middot; <b>${nRatingDerived}</b> from ratings` : "";
  $("stats").innerHTML = `<b>${state.movies.length}</b> movies &middot; <b>${nDecisive}</b> comparisons${ratingNote}${connNote}${tauNote}`;

  const list = $("top20");
  list.innerHTML = "";
  ranked.slice(0, 20).forEach((m, i) => {
    const li = document.createElement("li");
    const score = (scores.get(m.id) ?? 0).toFixed(3);
    li.innerHTML = `<span><span class="n">${i + 1}.</span>${m.title}${m.year ? ` (${m.year})` : ""}</span><span class="sc">${score}</span>`;
    list.appendChild(li);
  });
}

/** O(n^2) Kendall's tau between two score maps over their shared movies --
 * fine at this scale (called only on a manual refit), and doesn't need to
 * match scipy's tau-b tie handling exactly, just to be a sane stability
 * signal. `prevScoresObj` is a plain object (as persisted); `newScores` a
 * Map. */
function kendallTau(prevScoresObj, newScores) {
  const prev = new Map(Object.entries(prevScoresObj));
  const shared = [...newScores.keys()].filter((id) => prev.has(id));
  if (shared.length < 2) return null;
  let concordant = 0;
  let discordant = 0;
  for (let i = 0; i < shared.length; i++) {
    for (let j = i + 1; j < shared.length; j++) {
      const signA = Math.sign(prev.get(shared[i]) - prev.get(shared[j]));
      const signB = Math.sign(newScores.get(shared[i]) - newScores.get(shared[j]));
      if (signA === 0 || signB === 0) continue;
      if (signA === signB) concordant += 1;
      else discordant += 1;
    }
  }
  const total = concordant + discordant;
  return total === 0 ? null : (concordant - discordant) / total;
}

async function triggerRefit() {
  const btn = $("refit-btn");
  btn.disabled = true;
  const originalText = btn.textContent;
  btn.textContent = "Fitting…";
  try {
    const t0 = performance.now();
    const scores = M.fitScores(state.movies, state.caches.pairWins);
    const prevRun = latestModelRun();
    const kendallTauVsPrev = prevRun ? kendallTau(prevRun.scores, scores) : null;
    let nComparisons = 0;
    for (const v of state.caches.decisivePairCounts.values()) nComparisons += v;
    const run = {
      timestamp: new Date().toISOString(),
      nComparisons,
      scores: Object.fromEntries(scores),
      kendallTauVsPrev,
    };
    run.id = await storage.addModelRun(run);
    state.modelRuns.push(run);
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    btn.textContent = `Done (${elapsed}s)`;
    loadStatus();
    refreshHistoryIfOpen();
    setTimeout(() => {
      btn.textContent = originalText;
    }, 2000);
  } finally {
    btn.disabled = false;
  }
}

async function triggerSeed() {
  const btn = $("seed-btn");
  btn.disabled = true;
  const originalText = btn.textContent;
  btn.textContent = "Seeding…";
  try {
    const existingPairKeys = new Set(state.caches.pairCounts.keys());
    const rows = M.seedRatingComparisons(state.movies, existingPairKeys);
    if (rows.length) {
      const ids = await storage.addComparisons(rows);
      rows.forEach((r, i) => {
        r.id = ids[i];
        state.comparisons.push(r);
      });
      rebuildCaches();
    }
    btn.textContent = rows.length > 0 ? `+${rows.length} added` : "Already up to date";
    loadStatus();
    refreshHistoryIfOpen();
    setTimeout(() => {
      btn.textContent = originalText;
    }, 2500);
  } finally {
    btn.disabled = false;
  }
}

// -- Locking a movie or a rating tier ------------------------------------

function updateLockUI() {
  const locked = state.lockedMovieId !== null || state.lockedRating !== null;
  $("lock-input").hidden = locked;
  $("lock-rating-select").hidden = locked;
  $("lock-active").hidden = !locked;
}

// -- Comparison history (view / edit / delete past comparisons) ---------
//
// Everything lives in one in-memory array for the whole session, so
// pagination is just filter+sort+slice -- no cursor needed (the server
// version's cursor exists to survive concurrent inserts between paginated
// SQL queries, which can't happen here: it's one array in one JS thread).

const HISTORY_PAGE_SIZE = 25;
let historyOffset = 0;
let historyMovieFilter = null;
let historySourceFilter = "";

function filteredHistory() {
  return state.comparisons
    .filter((c) => historyMovieFilter == null || c.movieAId === historyMovieFilter || c.movieBId === historyMovieFilter)
    .filter((c) => historySourceFilter === "" || c.source === historySourceFilter)
    .sort((a, b) => b.id - a.id);
}

function makeHistBtn(text, active, danger) {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = text;
  b.title = text;
  b.className = "hist-btn" + (active ? " active" : "") + (danger ? " danger" : "");
  return b;
}

async function setComparisonWinner(c, winnerId) {
  c.winnerId = winnerId;
  await storage.putComparison(c);
  rebuildCaches();
  loadStatus();
  resetAndLoadHistory();
}

async function deleteComparisonRow(c) {
  state.comparisons = state.comparisons.filter((x) => x.id !== c.id);
  await storage.deleteComparison(c.id);
  rebuildCaches();
  loadStatus();
  resetAndLoadHistory();
}

function renderHistoryRow(c) {
  const li = document.createElement("li");
  li.className = "history-row";
  const movieA = state.moviesById.get(c.movieAId);
  const movieB = state.moviesById.get(c.movieBId);
  if (!movieA || !movieB) return li; // Orphaned row (shouldn't happen); render nothing.

  const label = document.createElement("span");
  label.className = "history-label";
  const aLabel = movieA.title + (movieA.year ? ` (${movieA.year})` : "");
  const bLabel = movieB.title + (movieB.year ? ` (${movieB.year})` : "");
  label.textContent = `${aLabel} vs ${bLabel}`;
  if (c.source === "rating") {
    const badge = document.createElement("span");
    badge.className = "history-source-badge";
    badge.textContent = "from rating";
    badge.title = "Auto-generated from your star ratings, not a comparison you made.";
    label.appendChild(badge);
  }
  const ts = document.createElement("span");
  ts.className = "history-timestamp";
  ts.textContent = new Date(c.timestamp).toLocaleString();
  label.appendChild(ts);

  const actions = document.createElement("span");
  actions.className = "history-actions";
  const btnA = makeHistBtn(aLabel, c.winnerId === movieA.id, false);
  const btnB = makeHistBtn(bLabel, c.winnerId === movieB.id, false);
  const btnSkip = makeHistBtn("skip", c.winnerId === null, false);
  const btnDelete = makeHistBtn("delete", false, true);
  btnA.addEventListener("click", () => setComparisonWinner(c, movieA.id));
  btnB.addEventListener("click", () => setComparisonWinner(c, movieB.id));
  btnSkip.addEventListener("click", () => setComparisonWinner(c, null));
  btnDelete.addEventListener("click", () => deleteComparisonRow(c));

  actions.append(btnA, btnB, btnSkip, btnDelete);
  li.append(label, actions);
  return li;
}

function loadHistoryPage() {
  const all = filteredHistory();
  const page = all.slice(historyOffset, historyOffset + HISTORY_PAGE_SIZE);
  const list = $("history-list");
  page.forEach((c) => list.appendChild(renderHistoryRow(c)));
  historyOffset += page.length;
  $("history-empty").hidden = list.children.length > 0;
  $("history-load-more").hidden = historyOffset >= all.length;
  $("history-count").textContent = all.length === 0 ? "" : `${list.children.length} of ${all.length} shown`;
}

function resetAndLoadHistory() {
  historyOffset = 0;
  $("history-list").innerHTML = "";
  loadHistoryPage();
}

function refreshHistoryIfOpen() {
  if (!$("history-body").hidden) resetAndLoadHistory();
}

function loadMovieDatalist() {
  const datalist = $("movie-datalist");
  datalist.innerHTML = "";
  movieLabelToId.clear();
  const sorted = [...state.movies].sort((a, b) => a.title.localeCompare(b.title) || (a.year ?? 0) - (b.year ?? 0));
  for (const m of sorted) {
    const label = m.year ? `${m.title} (${m.year})` : m.title;
    movieLabelToId.set(label, m.id);
    const opt = document.createElement("option");
    opt.value = label;
    datalist.appendChild(opt);
  }
}

// -- Setup panel: CSV sync, backup export/import, migration, TMDB key ---

let setupResultTimer = null;
function setSetupResult(msg) {
  const el = $("setup-result");
  el.textContent = msg;
  clearTimeout(setupResultTimer);
  setupResultTimer = setTimeout(() => {
    el.textContent = "";
  }, 8000);
}

async function triggerSync() {
  const diaryFile = $("diary-file").files[0];
  const ratingsFile = $("ratings-file").files[0];
  if (!diaryFile && !ratingsFile) {
    setSetupResult("Choose at least one CSV file first.");
    return;
  }
  const btn = $("sync-btn");
  btn.disabled = true;
  const originalText = btn.textContent;
  btn.textContent = "Syncing…";
  try {
    const diaryText = diaryFile ? await diaryFile.text() : null;
    const ratingsText = ratingsFile ? await ratingsFile.text() : null;
    const parsed = ingestText(diaryText, ratingsText);
    let nNew = 0;
    let nUpdated = 0;
    const toWrite = [];
    for (const rec of parsed) {
      const existing = state.moviesById.get(rec.id);
      if (existing) {
        nUpdated += 1;
        // A later sync missing a rating/watched-date (e.g. only diary.csv
        // this time) shouldn't blank out a value we already knew.
        if (rec.rating != null) existing.rating = rec.rating;
        if (rec.watchedDate != null) existing.watchedDate = rec.watchedDate;
        toWrite.push(existing);
      } else {
        nNew += 1;
        state.movies.push(rec);
        state.moviesById.set(rec.id, rec);
        toWrite.push(rec);
      }
    }
    await storage.putMovies(toWrite);
    rebuildCaches();
    loadStatus();
    loadMovieDatalist();
    if (state.movies.length >= 2) loadNextPair();
    setSetupResult(`Synced: ${nNew} new, ${nUpdated} updated (${state.movies.length} movies total).`);
  } catch (e) {
    setSetupResult(`Sync failed: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = originalText;
  }
}

function downloadBlob(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function csvEscape(v) {
  if (v == null) return "";
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function exportRankingCsv() {
  const scores = M.currentScores(state.movies, latestModelRun());
  const counts = state.caches.movieCounts;
  const ranked = [...state.movies].sort((a, b) => (scores.get(b.id) ?? 0) - (scores.get(a.id) ?? 0));
  const lines = ["rank,title,year,score,letterboxd_rating,n_comparisons"];
  ranked.forEach((m, i) => {
    const row = [i + 1, m.title, m.year ?? "", (scores.get(m.id) ?? 0).toFixed(4), m.rating ?? "", counts.get(m.id) ?? 0];
    lines.push(row.map(csvEscape).join(","));
  });
  downloadBlob("ranking.csv", lines.join("\n"), "text/csv");
}

function exportBackup() {
  const payload = {
    format: "letterboxd-ranker-backup-v1",
    exportedAt: new Date().toISOString(),
    movies: state.movies,
    comparisons: state.comparisons,
    modelRuns: state.modelRuns,
  };
  downloadBlob(`letterboxd-ranker-backup-${new Date().toISOString().slice(0, 10)}.json`, JSON.stringify(payload), "application/json");
}

async function reloadStateFromStorage() {
  const { movies, comparisons, modelRuns, meta } = await storage.loadAll();
  state.movies = movies;
  state.moviesById = new Map(movies.map((m) => [m.id, m]));
  state.comparisons = comparisons;
  state.modelRuns = modelRuns;
  state.meta = meta;
  rebuildCaches();
  loadMovieDatalist();
  loadStatus();
  state.current = null;
  loadNextPair();
  refreshHistoryIfOpen();
}

async function importBackup(file) {
  let data;
  try {
    data = JSON.parse(await file.text());
  } catch (e) {
    setSetupResult("That file isn't valid JSON.");
    return;
  }
  if (data.format !== "letterboxd-ranker-backup-v1") {
    setSetupResult("That doesn't look like a Letterboxd Ranker backup file.");
    return;
  }
  const nMovies = data.movies?.length ?? 0;
  const nComparisons = data.comparisons?.length ?? 0;
  if (!confirm(`This replaces everything currently in this browser with the backup (${nMovies} movies, ${nComparisons} comparisons). Continue?`)) return;
  await storage.clearAll();
  if (data.movies?.length) await storage.putMovies(data.movies);
  if (data.comparisons?.length) await storage.addComparisons(data.comparisons);
  for (const run of data.modelRuns ?? []) await storage.addModelRun(run);
  await reloadStateFromStorage();
  setSetupResult(`Imported ${nMovies} movies and ${nComparisons} comparisons.`);
}

/** Import a one-time comparisons export from the local server app (see
 * README's "Migrating to the static version"). Movies are matched by
 * title+year, not id (ids aren't portable between the two apps); a
 * comparison whose movie hasn't been synced here yet, or whose pair
 * already has a comparison recorded, is skipped -- safe to run more than
 * once. */
async function importMigration(file) {
  let data;
  try {
    data = JSON.parse(await file.text());
  } catch (e) {
    setSetupResult("That file isn't valid JSON.");
    return;
  }
  if (data.format !== "letterboxd-ranker-comparisons-v1") {
    setSetupResult("That doesn't look like a comparisons migration file.");
    return;
  }
  const existingPairKeys = new Set(state.caches.pairCounts.keys());
  const rowsToAdd = [];
  let nSkippedExisting = 0;
  let nSkippedUnknown = 0;
  for (const c of data.comparisons ?? []) {
    const aId = M.movieKey(c.movieA.title, c.movieA.year);
    const bId = M.movieKey(c.movieB.title, c.movieB.year);
    if (!state.moviesById.has(aId) || !state.moviesById.has(bId)) {
      nSkippedUnknown += 1;
      continue;
    }
    const key = M.pairKey(aId, bId);
    if (existingPairKeys.has(key)) {
      nSkippedExisting += 1;
      continue;
    }
    existingPairKeys.add(key); // Don't double-add a duplicate within this same file.
    const winnerId = c.winner === "a" ? aId : c.winner === "b" ? bId : null;
    rowsToAdd.push({ movieAId: aId, movieBId: bId, winnerId, timestamp: c.timestamp ?? new Date().toISOString(), source: "user" });
  }
  if (rowsToAdd.length) {
    const ids = await storage.addComparisons(rowsToAdd);
    rowsToAdd.forEach((r, i) => {
      r.id = ids[i];
      state.comparisons.push(r);
    });
    rebuildCaches();
    loadStatus();
    refreshHistoryIfOpen();
  }
  setSetupResult(
    `Migrated ${rowsToAdd.length} comparisons (${nSkippedExisting} already present, ${nSkippedUnknown} referenced a movie not yet synced here).`
  );
}

// -- Wiring ---------------------------------------------------------------

function wireEvents() {
  $("card-a").addEventListener("click", () => state.current && submitChoice(state.current.a.id));
  $("card-b").addEventListener("click", () => state.current && submitChoice(state.current.b.id));
  $("skip-btn").addEventListener("click", () => state.current && submitChoice(null));
  document.addEventListener("keydown", (e) => {
    if (!state.current) return;
    if (e.key === "ArrowLeft") submitChoice(state.current.a.id);
    else if (e.key === "ArrowRight") submitChoice(state.current.b.id);
    else if (e.key === " ") {
      e.preventDefault();
      submitChoice(null);
    }
  });

  $("refit-btn").addEventListener("click", triggerRefit);
  $("seed-btn").addEventListener("click", triggerSeed);

  const lockInput = $("lock-input");
  lockInput.addEventListener("input", () => {
    const label = lockInput.value.trim();
    if (!movieLabelToId.has(label)) return;
    state.lockedMovieId = movieLabelToId.get(label);
    state.lockedRating = null;
    $("lock-title").textContent = label;
    lockInput.value = "";
    updateLockUI();
    loadNextPair();
  });

  const lockRatingSelect = $("lock-rating-select");
  lockRatingSelect.addEventListener("change", () => {
    if (!lockRatingSelect.value) return;
    state.lockedRating = parseFloat(lockRatingSelect.value);
    state.lockedMovieId = null;
    $("lock-title").textContent = `${fmtRating(state.lockedRating)} tier`;
    lockRatingSelect.value = "";
    updateLockUI();
    loadNextPair();
  });

  $("unlock-btn").addEventListener("click", () => {
    state.lockedMovieId = null;
    state.lockedRating = null;
    updateLockUI();
    loadNextPair();
  });

  $("history-toggle").addEventListener("click", () => {
    const body = $("history-body");
    body.hidden = !body.hidden;
    $("history-toggle").textContent = body.hidden ? "show" : "hide";
    if (!body.hidden) resetAndLoadHistory();
  });
  $("history-load-more").addEventListener("click", loadHistoryPage);

  const historyFilterInput = $("history-filter-input");
  historyFilterInput.addEventListener("input", () => {
    const label = historyFilterInput.value.trim();
    const id = movieLabelToId.has(label) ? movieLabelToId.get(label) : null;
    if (id === historyMovieFilter) return;
    historyMovieFilter = id;
    resetAndLoadHistory();
  });
  $("history-filter-clear").addEventListener("click", () => {
    historyFilterInput.value = "";
    if (historyMovieFilter === null) return;
    historyMovieFilter = null;
    resetAndLoadHistory();
  });
  $("history-source-filter").addEventListener("change", (e) => {
    historySourceFilter = e.target.value;
    resetAndLoadHistory();
  });

  $("export-ranking-btn").addEventListener("click", exportRankingCsv);

  $("setup-toggle").addEventListener("click", () => {
    const body = $("setup-body");
    body.hidden = !body.hidden;
    $("setup-toggle").textContent = body.hidden ? "show" : "hide";
  });
  $("sync-btn").addEventListener("click", triggerSync);
  $("export-backup-btn").addEventListener("click", exportBackup);
  $("import-backup-input").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) importBackup(file);
    e.target.value = "";
  });
  $("import-migration-input").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) importMigration(file);
    e.target.value = "";
  });
  $("tmdb-key-save-btn").addEventListener("click", async () => {
    const key = $("tmdb-key-input").value.trim();
    state.meta.set("tmdbApiKey", key);
    await storage.setMeta("tmdbApiKey", key);
    setSetupResult(key ? "TMDB key saved." : "TMDB key cleared.");
  });
}

async function init() {
  wireEvents();
  await reloadStateFromStorage();
  $("tmdb-key-input").value = state.meta.get("tmdbApiKey") ?? "";
  if (state.movies.length === 0) {
    $("setup-body").hidden = false;
    $("setup-toggle").textContent = "hide";
  } else {
    $("setup-body").hidden = true;
    $("setup-toggle").textContent = "show";
  }
}

init();
