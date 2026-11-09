"""
Download Last.fm listener tags for the songs in the dataset, once, into tags.csv.

Why: the genres in artists.zip belong to the ARTIST, so every Brandy song is
labelled the same. Last.fm tags are per SONG and also describe mood and vocals
("aggressive", "mellow", "screamo"), which Spotify's numbers cannot measure.

The Streamlit app never calls Last.fm. It only reads the tags.csv this produces.

Setup (once):
    1. Get a free API key at https://www.last.fm/api/account/create
    2. Put it in the environment, NOT in the code:
         PowerShell:  $env:LASTFM_API_KEY = "your_key_here"
         Mac/Linux:   export LASTFM_API_KEY=your_key_here

Run:
    python fetch_tags.py --limit 50          # quick try, ~15 seconds
    python fetch_tags.py --min-popularity 50 # the real run, ~4 hours

Stopping it with Ctrl+C is safe: rerun the same command and it carries on from
where it stopped, because every row is written as soon as it arrives.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from recommender import artist_names

API_URL = "https://ws.audioscrobbler.com/2.0/"

# Last.fm asks for at most 5 requests per second. 0.25s between calls is 4/s.
DEFAULT_SLEEP = 0.25

# Tag counts run 0-100 and mean "how many listeners agreed". Low counts are noise
# (one person tagging a song "my favourite"), so keep only tags people agreed on.
MIN_TAG_COUNT = 10
MAX_TAGS = 10

TAG_SEPARATOR = "|"


class LastFmClient:
    """Small wrapper around the Last.fm API: one method per thing we ask for."""

    def __init__(self, api_key, base_url=API_URL, session=None, sleep_seconds=DEFAULT_SLEEP):
        self.api_key = api_key
        self.base_url = base_url
        self.session = session or requests.Session()
        self.sleep_seconds = sleep_seconds
        self._artist_cache = {}  # many songs share an artist; ask about each one only once

    def _get(self, params, retries=3):
        """One API call. Retries on rate limiting and server errors, then gives up."""
        params = {**params, "api_key": self.api_key, "format": "json", "autocorrect": "1"}
        for attempt in range(retries):
            if self.sleep_seconds:
                time.sleep(self.sleep_seconds)
            try:
                response = self.session.get(self.base_url, params=params, timeout=15)
            except requests.RequestException:
                continue  # network hiccup: try again
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    return None
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)  # wait 1s, then 2s, then 4s
                continue
            return None  # 400/403 etc: a retry would fail the same way
        return None

    @staticmethod
    def _parse_tags(payload, key, min_count=MIN_TAG_COUNT, max_tags=MAX_TAGS):
        """Pull the tag names out of Last.fm's JSON, which has a few shapes."""
        if not payload or "error" in payload:
            return []
        tags = (payload.get(key) or {}).get("tag") or []
        if isinstance(tags, dict):  # a single tag comes back as one object, not a list
            tags = [tags]
        names = []
        for tag in tags:
            try:
                count = int(tag.get("count", 0))
            except (TypeError, ValueError):
                count = 0
            name = str(tag.get("name", "")).strip().lower()
            if name and count >= min_count:
                names.append(name)
        return names[:max_tags]

    def track_tags(self, artist, track):
        payload = self._get({"method": "track.getTopTags", "artist": artist, "track": track})
        return self._parse_tags(payload, "toptags")

    def artist_tags(self, artist):
        if artist not in self._artist_cache:
            payload = self._get({"method": "artist.getTopTags", "artist": artist})
            self._artist_cache[artist] = self._parse_tags(payload, "toptags")
        return self._artist_cache[artist]

    def tags_for(self, artist, track):
        """Song tags if the song has any, otherwise the artist's. Returns (tags, source)."""
        tags = self.track_tags(artist, track)
        if tags:
            return tags, "track"
        tags = self.artist_tags(artist)
        if tags:
            return tags, "artist"
        return [], "none"


def load_targets(tracks_path="tracks.zip", min_popularity=50, limit=None):
    """The songs worth asking about: most popular version of each song+artist pair."""
    tracks = pd.read_csv(tracks_path, usecols=["id", "name", "artists", "popularity"])
    tracks = tracks.dropna(subset=["name", "artists"])
    tracks = tracks[tracks["popularity"] >= min_popularity]
    tracks = tracks.sort_values("popularity", ascending=False)
    tracks = tracks.drop_duplicates(subset=["name", "artists"])
    tracks["artist"] = tracks["artists"].map(lambda a: artist_names(a).split(",")[0].strip())
    targets = tracks[["id", "name", "artist"]].reset_index(drop=True)
    return targets.head(limit) if limit else targets


def already_done(out_path):
    """Ids already in tags.csv, so a rerun continues instead of starting over."""
    path = Path(out_path)
    if not path.exists():
        return set()
    try:
        return set(pd.read_csv(path, usecols=["id"])["id"].astype(str))
    except (ValueError, pd.errors.EmptyDataError):
        return set()


def fetch_all(targets, client, out_path="tags.csv", progress_every=50, log=print):
    """Ask Last.fm about each song and append it to the CSV as soon as it answers."""
    done = already_done(out_path)
    todo = targets[~targets["id"].astype(str).isin(done)]
    if done:
        log(f"{len(done):,} already saved, {len(todo):,} to go.")

    is_new_file = not Path(out_path).exists()
    counts = {"track": 0, "artist": 0, "none": 0}
    started = time.time()

    with open(out_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new_file:
            writer.writerow(["id", "tags", "source"])
        for position, row in enumerate(todo.itertuples(index=False), start=1):
            tags, source = client.tags_for(row.artist, row.name)
            counts[source] += 1
            writer.writerow([row.id, TAG_SEPARATOR.join(tags), source])
            handle.flush()  # so Ctrl+C never loses what was already fetched
            if position % progress_every == 0 or position == len(todo):
                rate = position / max(time.time() - started, 1e-9)
                left = (len(todo) - position) / rate / 60
                log(f"  {position:,}/{len(todo):,}  "
                    f"song tags: {counts['track']:,}  artist tags: {counts['artist']:,}  "
                    f"nothing: {counts['none']:,}  ~{left:.0f} min left")
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download Last.fm tags into tags.csv")
    parser.add_argument("--min-popularity", type=int, default=50,
                        help="skip songs below this Spotify popularity (default 50, ~66k songs)")
    parser.add_argument("--limit", type=int, help="only the first N songs, for a quick try")
    parser.add_argument("--out", default="tags.csv")
    parser.add_argument("--tracks", default="tracks.zip")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                        help="seconds between requests (default 0.25 = 4 per second)")
    parser.add_argument("--base-url", default=API_URL, help="override the API address (for testing)")
    args = parser.parse_args(argv)

    api_key = os.environ.get("LASTFM_API_KEY")
    if not api_key:
        print("No API key found. Set LASTFM_API_KEY first:\n"
              '  PowerShell:  $env:LASTFM_API_KEY = "your_key_here"\n'
              "  Mac/Linux:   export LASTFM_API_KEY=your_key_here", file=sys.stderr)
        return 1

    targets = load_targets(args.tracks, args.min_popularity, args.limit)
    print(f"{len(targets):,} songs to look up. Writing to {args.out}. Ctrl+C is safe.")
    client = LastFmClient(api_key, base_url=args.base_url, sleep_seconds=args.sleep)
    try:
        counts = fetch_all(targets, client, args.out)
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to carry on.")
        return 0
    print(f"Done. song tags: {counts['track']:,}  artist tags: {counts['artist']:,}  "
          f"nothing: {counts['none']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
