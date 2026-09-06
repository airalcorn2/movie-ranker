"""Optional TMDB poster lookup.

Only activates if a TMDB_API_KEY env var is set. Free to obtain at
https://www.themoviedb.org/settings/api. If unset, callers just get None
and the UI falls back to a plain title card.
"""

import os

import requests

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
IMAGE_BASE = "https://image.tmdb.org/t/p/w342"


def fetch_poster_url(title: str, year: int | None) -> str | None:
    """The poster image URL for the best TMDB match, or None if no key is
    set, nothing matches, or the request fails."""
    if not TMDB_API_KEY:
        return None
    params = {"api_key": TMDB_API_KEY, "query": title}
    if year:
        params["year"] = str(year)
    try:
        resp = requests.get(SEARCH_URL, params=params, timeout=5)
        resp.raise_for_status()
        results = resp.json().get("results") or []
    except requests.RequestException:
        return None
    if not results:
        return None
    poster_path = results[0].get("poster_path")
    if not poster_path:
        return None
    return IMAGE_BASE + poster_path
