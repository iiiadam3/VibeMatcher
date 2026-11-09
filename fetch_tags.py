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
import re
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

# Last.fm error codes that mean "try again", not "this song does not exist".
# 8 = operation failed, 11 = service offline, 16 = temporary error, 29 = rate limit exceeded.
# (6 = "Track not found" is a real answer and is NOT retried.)
RETRYABLE_ERRORS = {8, 11, 16, 29}


# Words that mark "this part of the title is not the song's real name".
# Last.fm lists a song once, so "Burning Down the House - 2005 Remaster" finds nothing.
NOISE_WORDS = (r"feat|ft|featuring|with|remaster(?:ed)?|version|edit|mix|remix|live|ao vivo|en vivo|"
               r"mono|stereo|deluxe|bonus|acoustic|demo|radio|single|album|explicit|clean|instrumental|"
               r"original|extended|reprise|session|recorded|\d{4}")
_NOISE = re.compile(rf"\b(?:{NOISE_WORDS})\b", re.IGNORECASE)
_BRACKETS = re.compile(r"\s*[\(\[][^\(\)\[\]]*[\)\]]")
_FEAT_TAIL = re.compile(r"\s+(?:feat|ft|featuring)\.?\s.*$", re.IGNORECASE)


def clean_title(title):
    """
    Strip the parts of a song title Last.fm doesn't use.

      "Burning Down the House - 2005 Remaster"  -> "Burning Down the House"
      "Space Cadet (feat. Gunna)"               -> "Space Cadet"
      "Empire (Let Them Sing)"                  -> unchanged (the brackets are the real title)
      "A - Ö"                                   -> unchanged
    """
    cleaned = _BRACKETS.sub(lambda m: "" if _NOISE.search(m.group(0)) else m.group(0), str(title))
    cleaned = _FEAT_TAIL.sub("", cleaned)
    # "Song - Something": drop the part after the LAST " - " only if it is noise.
    # Repeat, because titles can stack endings: "Song - Remastered 2011 - Mono".
    while True:
        head, sep, tail = cleaned.rpartition(" - ")
        if not (sep and head.strip() and _NOISE.search(tail)):
            break
        cleaned = head
    cleaned = cleaned.strip()
    return cleaned or str(title).strip()  # never return an empty title


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
                    payload = response.json()
                except ValueError:
                    return None
                if isinstance(payload, dict) and payload.get("error") in RETRYABLE_ERRORS:
                    time.sleep(2 ** attempt)
                    continue
                return payload
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

    def diagnose_track(self, artist, track, min_count=MIN_TAG_COUNT):
        """
        One lookup, with the reason spelled out. Used by --diagnose, never by the real run.
        Returns (verdict, detail) where verdict is OK, FILTERED, MISS or ERROR.
        """
        params = {"method": "track.getTopTags", "artist": artist, "track": track,
                  "api_key": self.api_key, "format": "json", "autocorrect": "1"}
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        try:
            response = self.session.get(self.base_url, params=params, timeout=15)
        except requests.RequestException as exc:
            return "ERROR", f"network problem: {type(exc).__name__}"
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if response.status_code != 200 and not (isinstance(payload, dict) and "error" in payload):
            return "ERROR", f"HTTP {response.status_code}"
        if not isinstance(payload, dict):
            return "ERROR", f"HTTP {response.status_code}, answer was not JSON"
        if "error" in payload:
            code = payload.get("error")
            kind = "ERROR" if code in RETRYABLE_ERRORS or code in (4, 10, 26) else "MISS"
            return kind, f"HTTP {response.status_code}, Last.fm error {code}: {payload.get('message', '')}"
        tags = (payload.get("toptags") or {}).get("tag") or []
        if isinstance(tags, dict):
            tags = [tags]
        counts = []
        for tag in tags:
            try:
                counts.append(int(tag.get("count", 0)))
            except (TypeError, ValueError):
                counts.append(0)
        kept = [c for c in counts if c >= min_count]
        if not counts:
            return "MISS", "Last.fm knows the song but it has no tags"
        if not kept:
            return "FILTERED", f"{len(counts)} tags, all below count {min_count} (highest is {max(counts)})"
        return "OK", f"{len(kept)} of {len(counts)} tags kept (top counts {counts[:4]})"

    def artist_tags(self, artist):
        if artist not in self._artist_cache:
            payload = self._get({"method": "artist.getTopTags", "artist": artist})
            self._artist_cache[artist] = self._parse_tags(payload, "toptags")
        return self._artist_cache[artist]

    def tags_for(self, artist, track):
        """
        Song tags if the song has any, otherwise the artist's. Returns (tags, source).

        Tries the cleaned title first ("Song"), then the title as written in the dataset
        ("Song - 2005 Remaster"), then falls back to the artist.
        """
        names = [clean_title(track)]
        if names[0] != track:
            names.append(track)
        for name in names:
            tags = self.track_tags(artist, name)
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


def diagnose(targets, client, how_many, log=print):
    """
    Print exactly what Last.fm answered for a RANDOM sample of songs, and tally the reasons.

    Random on purpose: the list is sorted by popularity, so its first rows are all brand-new
    hits, which have had little time to collect tags and would give a misleadingly bad picture.
    """
    tally = {}
    sample = targets.sample(min(how_many, len(targets)), random_state=0)
    for position, row in enumerate(sample.itertuples(index=False), start=1):
        sent = clean_title(row.name)
        verdict, detail = client.diagnose_track(row.artist, sent)
        tally[verdict] = tally.get(verdict, 0) + 1
        log(f"{position:>3}. {verdict:<8} {row.artist} - {sent}  ->  {detail}")
    log("\nSummary: " + ", ".join(f"{k} {v}" for k, v in sorted(tally.items())))
    return tally


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download Last.fm tags into tags.csv")
    parser.add_argument("--min-popularity", type=int, default=50,
                        help="skip songs below this Spotify popularity (default 50, ~66k songs)")
    parser.add_argument("--limit", type=int, help="only the first N songs, for a quick try")
    parser.add_argument("--out", default="tags.csv")
    parser.add_argument("--tracks", default="tracks.zip")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                        help="seconds between requests (default 0.25 = 4 per second)")
    parser.add_argument("--diagnose", type=int, metavar="N",
                        help="don't save anything: show what Last.fm answers for the first N songs")
    parser.add_argument("--base-url", default=API_URL, help="override the API address (for testing)")
    args = parser.parse_args(argv)

    api_key = os.environ.get("LASTFM_API_KEY")
    if not api_key:
        print("No API key found. Set LASTFM_API_KEY first:\n"
              '  PowerShell:  $env:LASTFM_API_KEY = "your_key_here"\n'
              "  Mac/Linux:   export LASTFM_API_KEY=your_key_here", file=sys.stderr)
        return 1

    targets = load_targets(args.tracks, args.min_popularity, args.limit)
    client = LastFmClient(api_key, base_url=args.base_url, sleep_seconds=args.sleep)
    if args.diagnose:
        diagnose(targets, client, args.diagnose)
        return 0
    print(f"{len(targets):,} songs to look up. Writing to {args.out}. Ctrl+C is safe.")
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
