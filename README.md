# Letterboxd Ranker

![User Interface](ui.png)

Rank your Letterboxd diary/ratings export via pairwise comparisons + a Bradley-Terry model (fit with [`choix`](https://github.com/lucasmaystre/choix)), with active learning picking which pairs to ask about so you don't have to compare every possible movie combination.

## How it works

1. `ingest.py` parses your Letterboxd CSV export into a local SQLite database.
2. `app.py` (FastAPI) serves a comparison UI at `/`: two movies, click the one you prefer (or skip).
3. Each comparison is stored, and every `REFIT_INTERVAL` (15) decisive comparisons the Bradley-Terry model is refit over all comparisons so far.
4. `/next_pair` picks the next pair to show you, in priority order:
   - bridge disconnected parts of the comparison graph first, so the ranking stays globally meaningful;
   - early on, sample movies with adjacent star ratings (avoids "obviously" one-sided comparisons while the model is cold);
   - once a model exists, sample the pair whose predicted win probability is closest to 50/50 (max information), weighted toward under-compared movies.
5. Scores start from your Letterboxd star ratings (movies without a rating start at the mean) and are pulled toward that prior via synthetic "pseudo-comparisons" between adjacent-rated movies.
   That pull fades out as real comparisons accumulate.
6. `/status` shows comparison counts, whether the comparison graph is fully connected, a live top-20, and a Kendall-tau stability score against the previous fit so you know when the ranking has settled.
7. `/export` downloads the full ranking as CSV.
8. Misclick, or changed your mind? The "Comparison history" panel at the bottom of the page lists past comparisons and lets you reassign the winner or delete the row entirely.
   Edits refit the model immediately rather than waiting for `REFIT_INTERVAL`.
   Type a movie's name into the filter box to see only comparisons involving it — handy after a rewatch, to revisit every past comparison for that film in one place.
   The list always reloads from scratch on open and after any edit, so it can't go stale if you keep comparing while it's open.

## Installing the Python dependencies

This assumes the `movie-ranker` pyenv virtualenv already active in this directory (see `.python-version`).
Any Python 3.10+ environment works too.

```bash
pip3 install -r requirements.txt
```

## Downloading your Letterboxd data

[Letterboxd → Settings → Data](https://letterboxd.com/settings/data/) → Export Your Data.
Unzip it and copy `diary.csv` and/or `ratings.csv` into `data/` (or point `ingest.py` directly at the files).
Both are supported and merged.
If a film appears in both, or has multiple diary entries (rewatches), the most recently dated entry wins.

```bash
python3 ingest.py --data-dir data/
# or explicitly:
python3 ingest.py --diary path/to/diary.csv --ratings path/to/ratings.csv
```

Re-run `ingest.py` any time to add new movies or update stars/watched dates.
It upserts by (title, year), so it's safe to run repeatedly.

## Running

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000 and start comparing.
Left/right arrow keys pick A/B, spacebar skips ("can't decide").

## Accessing from your phone (same Wi-Fi network)

Bind to all interfaces instead of just localhost:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Find this machine's LAN IP:

```bash
hostname -I
```

If a firewall is active on this machine (e.g. Fedora's `firewalld`), open the port for the current boot:

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

## Files

- `db.py` — SQLite schema (`movies`, `comparisons`, `model_runs`) + connection helper
- `ingest.py` — CSV export → SQLite
- `model.py` — Bradley-Terry fitting (via `choix`) + active-learning pair selection
- `posters.py` — optional TMDB poster lookup
- `app.py` — FastAPI server: `/`, `/status`, `/next_pair`, `/compare`, `/comparisons` (list/edit/delete, filterable by movie), `/movies`, `/export`
- `static/index.html` — the comparison UI (plain HTML/JS, no build step)
