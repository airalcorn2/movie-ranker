# Letterboxd Ranker

![User Interface](ui.png)

Rank your Letterboxd diary/ratings export via pairwise comparisons + a [Bradley-Terry model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) (fit with [`choix`](https://github.com/lucasmaystre/choix)), with active learning picking which pairs to ask about so you don't have to compare every possible movie combination.

There are two ways to run this, sharing the same design and UI:

- **A local server** (`app.py` + SQLite) -- this section describes it.
  Run it on your own machine; see [Static version (GitHub Pages, no server)](#static-version-github-pages-no-server) below for the other option.
- **A static, no-server version** (`docs/`) -- host it for free on GitHub Pages, with all data kept in your browser instead of a server-side database.
  No Python needed to run it at all.

## How it works

1. Export your Letterboxd data, drop `diary.csv`/`ratings.csv` into `data/`, then click "Sync from data/" in the header (or run `ingest.py` from the command line -- same code either way) to parse them into a local SQLite database.
2. `app.py` (FastAPI) serves a comparison UI at `/`: two movies, click the one you prefer (or skip).
3. Each comparison is stored.
   Fitting is manual: click "Refit now" (top right) whenever you want the ranking to reflect what you've compared so far.
4. `/next_pair` picks the next pair to show you, in priority order:
   - if you've locked a movie or a rating tier (see below), pick from there and ignore everything else until you unlock it;
   - otherwise, bridge disconnected parts of the comparison graph first, so the ranking stays globally meaningful;
   - early on, sample movies with the same star rating (cross-tier order is already predictable from the rating, so same-tier pairs are where a comparison is actually uncertain);
   - once a model exists, sample the pair whose predicted win probability is closest to 50/50 (max information), weighted toward under-compared movies.
   Just rewatched something and want its ranking to converge fast?
   Type its title into the "🔒 Lock a movie" box in the header — every pair from then on will include it, chosen by uncertainty, until you hit "Unlock".
   Want to sort out one whole star tier instead (e.g., settle the order of everything you rated 4 stars) — use the "🔒★ Lock a rating" dropdown next to it; every pair is then drawn from that tier alone, most-uncertain first.
   The two locks are mutually exclusive: picking one clears the other.
5. Movies without a rating start at the mean.
   Click "Seed from ratings" to additionally insert one *permanent* comparison for every pair of differently-rated movies (higher rating wins) — this is a real, stored comparison (`source: "rating"`), weighted identically to one of your own clicks, not a temporary nudge that fades away.
   It never touches same-rating pairs, so within-tier order (where the real uncertainty is) is always left for your own comparisons.
   Change your mind about a specific one?
   It's just a normal row in the Comparison history panel — reassign or delete it there like anything else.
   Safe to click again any time (e.g., after syncing newly-rated movies): it only fills in pairs that don't already have a comparison recorded -- which means re-rating a movie *after* it's already been seeded against something won't update that old comparison, since one already exists for the pair.
   Fix it the same way as a misclick: filter the Comparison history panel by that movie, delete the stale rating-derived row, then click "Seed from ratings" again.
6. `/status` shows comparison counts (yours, separate from the rating-derived count), whether the comparison graph is fully connected, a live top-20, and a Kendall-tau stability score against the previous fit so you know when the ranking has settled.
7. `/export` downloads the full ranking as CSV.
8. Misclick, or changed your mind?
   The "Comparison history" panel at the bottom of the page lists past comparisons and lets you reassign the winner or delete the row entirely — filter by movie (handy after a rewatch) or by source (yours vs. rating-derived).
   The list always reloads from scratch on open and after any edit, so it can't go stale if you keep comparing while it's open.
9. Nothing above refits automatically — click "Refit now" (top right) whenever you want the ranking to reflect what's changed.
   `/next_pair` and `/status` stay fast even with 100K+ comparisons in the table (from seeding) because the comparison graph is cached in memory and updated incrementally, not rescanned on every click; only an explicit refit pays the O(n²) fitting cost (a few seconds at ~700 movies).

## Installing the Python dependencies

This assumes the `movie-ranker` pyenv virtualenv already active in this directory (see `.python-version`).
Any Python 3.10+ environment works too.

```bash
pip3 install -r requirements.txt
```

## Downloading your Letterboxd data

[Letterboxd → Settings → Data](https://letterboxd.com/settings/data/) → Export Your Data.
Unzip it and copy `diary.csv` and/or `ratings.csv` into `data/`.
Both are supported and merged.
If a film appears in both, or has multiple diary entries (rewatches), the most recently dated entry wins.

Once the files are in `data/`, click "Sync from data/" in the header (see [Running](#running) below) -- it re-reads both files and upserts movies, no command line needed.
The command-line form still works too, and accepts explicit paths instead of `data/`:

```bash
python3 ingest.py --data-dir data/
# or explicitly:
python3 ingest.py --diary path/to/diary.csv --ratings path/to/ratings.csv
```

Re-sync (button or script) any time to add new movies or update stars/watched dates.
It upserts by (title, year), so it's safe to run repeatedly.

## Running

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000 and start comparing.
Left/right arrow keys pick A/B, spacebar skips ("can't decide").

## Stopping the app

If it's running in the foreground of the terminal you started it in, press Ctrl+C.

If it's in the background (started with `&`, `nohup`, or you've just lost track of which terminal it's in), find and stop it by its command line rather than guessing a PID:

```bash
pkill -f "uvicorn app:app"
```

Or see it before killing it:

```bash
pgrep -fal "uvicorn app:app"
kill <pid>
```

`--reload` runs as a supervisor process managing a separate worker process underneath it; stopping the supervisor (the PID either command above finds) shuts the worker down with it, so nothing's left running.

## Accessing from your phone (same Wi-Fi network)

Bind to all interfaces instead of just localhost:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Find this machine's LAN IP:

```bash
hostname -I
```

If a firewall is active on this machine (e.g., Fedora's `firewalld`), open the port for the current boot:

```bash
sudo firewall-cmd --add-port=8000/tcp
```

or permanently:

```bash
sudo firewall-cmd --add-port=8000/tcp --permanent && sudo firewall-cmd --reload
```

Then, on your phone (connected to the same Wi-Fi, not cellular), visit `http://<that-ip>:8000`.
The IP is usually DHCP-assigned and can change on reconnect; re-run `hostname -I` if the page stops loading, or set a DHCP reservation in your router for a stable address.

## Optional: posters via TMDB

Set a [TMDB API key](https://www.themoviedb.org/settings/api) (free) to show posters in the comparison UI instead of plain title cards:

```bash
export TMDB_API_KEY=your_key_here
uvicorn app:app --reload
```

Without a key set, the UI just shows title/year/rating cards.
Everything else works the same.

## Static version (GitHub Pages, no server)

Everything above (`app.py`, SQLite, `ingest.py`) has a client-side-only counterpart in `docs/`: same UI and the same active-learning ideas (locks, seeding, connectivity-first sampling, manual refit), but with zero server -- CSV parsing, Bradley-Terry fitting, and pair selection all run in the browser, and the data lives in the browser's IndexedDB instead of a SQLite file.
That makes it a genuinely good fit for free static hosting: GitHub Pages never sees or stores any of your data, only the app's code, so there's no ephemeral-storage risk the way there would be with a free *server* host.

It's an independent reimplementation, not a port -- in particular, the Bradley-Terry fit uses the standard MM algorithm (Hunter 2004 / Zermelo's algorithm) instead of `choix`'s Markov-chain formulation, since that needs only plain array passes, no linear-algebra library.
Both converge to the same underlying MLE and are regularized in the same *spirit*, so rankings should be similar in practice, but the two versions won't produce bit-identical scores from the same comparisons.

### Hosting it

1. Push this repo (or just `docs/`) to a **public** GitHub repository -- GitHub Pages' free tier requires that, but it's only your *code* that's public, never your data (nothing you compare ever leaves your browser, except optional poster lookups to TMDB).
2. Repo Settings → Pages → Source: "Deploy from a branch" → Branch: `main`, folder `/docs` → Save.
3. GitHub gives you a URL like `https://<username>.github.io/<repo>/` within a minute or two.
   That's the whole app.

### Getting your data in

There's no `data/` folder to drop files into here -- use the "Data & setup" panel at the top of the page (expanded automatically the first time, since there's nothing to rank yet):

1. Export from [Letterboxd → Settings → Data](https://letterboxd.com/settings/data/), unzip it, and pick `diary.csv`/`ratings.csv` in the panel's file inputs, then click Sync.
   Re-sync any time the same way (safe to run repeatedly -- upserts by title+year, exactly like `ingest.py`).
2. Click "Seed from ratings" if you want your star ratings to count as permanent comparisons, same as the server version.
3. Optionally paste a free [TMDB API key](https://www.themoviedb.org/settings/api) for posters -- stored only in your browser, sent only to TMDB.

### Moving between devices

This browser is the only copy of your data -- there's no server for a second device to talk to.
The "Data & setup" panel has an **Export backup (.json)** button (all your movies, comparisons, and fit history in one file) and a matching **Import backup (.json)** to load it back in on another device or browser.
Import replaces everything currently in that browser, so it's meant for "set up a new device" or "restore," not merging two independent histories.
Worth exporting a backup periodically regardless, purely as a safety copy -- IndexedDB is easier to lose by accident (clearing browsing data, a private window, a wiped machine) than a file you can just copy somewhere yourself.

### Migrating to the static version

If you've been using the local server version and want to bring your real comparisons over: open it, click "↓ Export comparisons for the static version" (near the CSV export link), sync the same `diary.csv`/`ratings.csv` on the static version first, then use "Import comparisons (.json)" in its Data & setup panel to load the file.
It matches movies by title+year (not by id -- ids aren't portable between the two apps) and only touches pairs that don't already have a comparison recorded, so it's safe to run again later.
Only your real comparisons are exported, not the rating-derived ones -- those regenerate instantly by clicking "Seed from ratings" in the static version instead.

### Known limits

Fitting is O(n²) per iteration (same asymptotic shape as the server version's `choix` call, different constant), comfortably fast for a few thousand movies but noticeably slower beyond that -- it runs on the main thread, so the tab will visibly pause for the duration of a refit on a very large collection.

## Files

- `db.py` — SQLite schema (`movies`, `comparisons`, `model_runs`) + connection helper
- `ingest.py` — CSV export → SQLite; the CLI entry point, and the code `app.py`'s `/sync` route calls into
- `model.py` — Bradley-Terry fitting (via `choix`) + active-learning pair selection
- `posters.py` — optional TMDB poster lookup
- `app.py` — FastAPI server: `/`, `/status`, `/next_pair` (optionally `?locked_movie_id=` or `?locked_rating=`), `/compare`, `/comparisons` (list/edit/delete, filterable by movie and/or source), `/movies`, `/sync`, `/refit`, `/seed_rating_comparisons`, `/export`, `/export_comparisons`
- `static/index.html` — the local-server comparison UI (plain HTML/JS, no build step)
- `docs/` — the static, no-server version (see [Static version (GitHub Pages, no server)](#static-version-github-pages-no-server)); the `docs` name is required by GitHub Pages' "deploy from `/docs`" option
  - `index.html` — markup + styling, structurally close to `static/index.html`
  - `app.js` — UI controller: wires up DOM events, holds the session's in-memory state
  - `model.js` — Bradley-Terry fitting (MM algorithm) + active-learning pair selection; the client-side counterpart to `model.py`
  - `storage.js` — IndexedDB persistence
  - `csv.js` — client-side diary.csv/ratings.csv parsing; the counterpart to `ingest.py`
