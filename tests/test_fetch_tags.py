"""
Tests for the Last.fm downloader. No real network calls: a fake session answers
with the JSON shapes Last.fm actually returns, including its odd ones.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fetch_tags import LastFmClient, already_done, clean_title, diagnose, fetch_all, load_targets  # noqa: E402


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """Answers from a dict of {(method, artist, track-or-None): payload}."""

    def __init__(self, answers, statuses=None):
        self.answers = answers
        self.statuses = statuses or {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        params = params or {}
        key = (params.get("method"), params.get("artist"), params.get("track"))
        self.calls.append(key)
        status = self.statuses.get(key, 200)
        return FakeResponse(self.answers.get(key), status)


def tags_payload(*pairs):
    return {"toptags": {"tag": [{"name": n, "count": c} for n, c in pairs]}}


def client_for(session):
    return LastFmClient("fake-key", session=session, sleep_seconds=0)


# ---------------------------------------------------------------------------
# Reading Last.fm's answers
# ---------------------------------------------------------------------------
def test_keeps_agreed_tags_and_drops_noise():
    session = FakeSession({("track.getTopTags", "Arctic Monkeys", "Do I Wanna Know?"):
                           tags_payload(("indie rock", 100), ("dark", 40), ("my favourite", 1))})
    assert client_for(session).track_tags("Arctic Monkeys", "Do I Wanna Know?") == ["indie rock", "dark"]


def test_single_tag_comes_back_as_an_object_not_a_list():
    session = FakeSession({("track.getTopTags", "A", "B"): {"toptags": {"tag": {"name": "punk", "count": 90}}}})
    assert client_for(session).track_tags("A", "B") == ["punk"]


@pytest.mark.parametrize("payload", [
    {"error": 6, "message": "Track not found"},
    {"toptags": {"tag": []}},
    {"toptags": {}},
    {},
    None,
])
def test_missing_or_broken_answers_give_no_tags(payload):
    session = FakeSession({("track.getTopTags", "A", "B"): payload})
    assert client_for(session).track_tags("A", "B") == []


def test_tag_list_is_capped():
    session = FakeSession({("track.getTopTags", "A", "B"): tags_payload(*[(f"t{i}", 100) for i in range(30)])})
    assert len(client_for(session).track_tags("A", "B")) == 10


# ---------------------------------------------------------------------------
# Falling back from song to artist
# ---------------------------------------------------------------------------
def test_song_tags_win_when_the_song_has_them():
    session = FakeSession({("track.getTopTags", "Brandy", "Come a Little Closer"): tags_payload(("r&b", 80)),
                           ("artist.getTopTags", "Brandy", None): tags_payload(("hip hop", 90))})
    assert client_for(session).tags_for("Brandy", "Come a Little Closer") == (["r&b"], "track")


def test_falls_back_to_the_artist_when_the_song_has_none():
    session = FakeSession({("track.getTopTags", "Brandy", "Rare B-side"): {"toptags": {"tag": []}},
                           ("artist.getTopTags", "Brandy", None): tags_payload(("r&b", 90))})
    assert client_for(session).tags_for("Brandy", "Rare B-side") == (["r&b"], "artist")


def test_source_is_none_when_nobody_tagged_anything():
    session = FakeSession({})
    assert client_for(session).tags_for("Nobody", "Nothing") == ([], "none")


def test_each_artist_is_asked_about_only_once():
    session = FakeSession({("artist.getTopTags", "Brandy", None): tags_payload(("r&b", 90))})
    client = client_for(session)
    client.tags_for("Brandy", "Song One")
    client.tags_for("Brandy", "Song Two")
    assert sum(1 for c in session.calls if c[0] == "artist.getTopTags") == 1


def test_server_errors_are_retried_then_given_up_on():
    key = ("track.getTopTags", "A", "B")
    session = FakeSession({}, statuses={key: 503})
    client = LastFmClient("k", session=session, sleep_seconds=0)
    client._get({"method": "track.getTopTags", "artist": "A", "track": "B"}, retries=3)
    assert len(session.calls) == 3


def test_client_sends_the_api_key_and_asks_for_json():
    session = FakeSession({})
    client_for(session).track_tags("A", "B")
    assert session.calls == [("track.getTopTags", "A", "B")]


# ---------------------------------------------------------------------------
# Choosing songs and writing the file
# ---------------------------------------------------------------------------
@pytest.fixture
def tracks_csv(tmp_path):
    path = tmp_path / "tracks.csv"
    pd.DataFrame([
        {"id": "35iwgR4jXetI318WEWsa1Q", "name": "Popular", "artists": "['Artist A', 'Guest']", "popularity": 90},
        {"id": "7hQJA50XrCWABAu5v6QZ4i", "name": "Popular", "artists": "['Artist A', 'Guest']", "popularity": 60},  # same song, worse copy
        {"id": "1oHNvJVbFkexQc0BpQp7Y4", "name": "Unknown", "artists": "['Artist B']", "popularity": 5},
        {"id": "5ihS6UUlyQAfmp48eSkxuQ", "name": "Middling", "artists": "['Artist C']", "popularity": 55},
    ]).to_csv(path, index=False)
    return path


def test_targets_skip_unpopular_and_duplicate_songs(tracks_csv):
    targets = load_targets(tracks_csv, min_popularity=50)
    assert targets["id"].tolist() == ["35iwgR4jXetI318WEWsa1Q", "5ihS6UUlyQAfmp48eSkxuQ"]


def test_targets_use_the_first_artist_name(tracks_csv):
    assert load_targets(tracks_csv, min_popularity=50)["artist"].iloc[0] == "Artist A"


def test_limit_shortens_the_list(tracks_csv):
    assert len(load_targets(tracks_csv, min_popularity=0, limit=2)) == 2


def test_writes_a_row_per_song_with_a_header(tmp_path, tracks_csv):
    out = tmp_path / "tags.csv"
    session = FakeSession({("track.getTopTags", "Artist A", "Popular"): tags_payload(("rock", 90), ("dark", 20))})
    fetch_all(load_targets(tracks_csv, min_popularity=50), client_for(session), out, log=lambda *a: None)
    written = pd.read_csv(out, keep_default_na=False)
    assert written.columns.tolist() == ["id", "tags", "source"]
    by_id = written.set_index("id")
    assert by_id.loc["35iwgR4jXetI318WEWsa1Q", "tags"] == "rock|dark"
    assert by_id.loc["5ihS6UUlyQAfmp48eSkxuQ", "source"] == "none"


def test_rerunning_skips_songs_already_saved(tmp_path, tracks_csv):
    out = tmp_path / "tags.csv"
    targets = load_targets(tracks_csv, min_popularity=50)
    session = FakeSession({})
    fetch_all(targets, client_for(session), out, log=lambda *a: None)
    first_pass = len(session.calls)

    session2 = FakeSession({})
    fetch_all(targets, client_for(session2), out, log=lambda *a: None)
    assert first_pass > 0 and session2.calls == []  # nothing left to ask about
    assert len(pd.read_csv(out)) == 2  # and no duplicate rows


def test_already_done_handles_a_missing_or_empty_file(tmp_path):
    assert already_done(tmp_path / "nope.csv") == set()
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert already_done(empty) == set()


# ---------------------------------------------------------------------------
# Cleaning song titles (real examples from the dataset)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("Burning Down the House - 2005 Remaster", "Burning Down the House"),
    ("One For Daddy-O - Remastered", "One For Daddy-O"),
    ("Stay (Remastered 2009)", "Stay"),
    ("Space Cadet (feat. Gunna)", "Space Cadet"),
    ("4 Minutes (feat. Justin Timberlake and Timbaland)", "4 Minutes"),
    ("hole in the bottle (with Shania Twain)", "hole in the bottle"),
    ("Dices (Remix) [feat. Arcangel & Wisin]", "Dices"),
    ("Can You Feel It - 7\" Version", "Can You Feel It"),
    ("Mudou a Estação - Ao Vivo", "Mudou a Estação"),
    ("Comando X (En Vivo)", "Comando X"),
    ("Is This Love - Live At The Pavillon De Paris, 1977", "Is This Love"),
    ("Hey Luv (Anything) (feat. 112)", "Hey Luv (Anything)"),
    ("Song - Remastered 2011 - Mono", "Song"),
    ("Sorry Not Sorry feat. Demi", "Sorry Not Sorry"),
])
def test_clean_title_removes_noise(raw, expected):
    assert clean_title(raw) == expected


@pytest.mark.parametrize("title", [
    "Empire (Let Them Sing)",           # brackets are part of the real title
    "(Our Love) Don't Throw It All Away",
    "A - Ö",                            # a real title with a dash
    "Live Forever",                     # 'live' as a word in the title, not a version tag
    "Mr. Brightside",
    "1999",
    "(Live)",                           # cleaning must never leave an empty title
])
def test_clean_title_leaves_real_titles_alone(title):
    assert clean_title(title) == title


def test_cleaned_title_is_tried_before_the_original():
    cleaned = ("track.getTopTags", "Talking Heads", "Burning Down the House")
    session = FakeSession({cleaned: tags_payload(("new wave", 90))})
    tags, source = client_for(session).tags_for("Talking Heads", "Burning Down the House - 2005 Remaster")
    assert (tags, source) == (["new wave"], "track")
    assert session.calls == [cleaned]  # found on the first ask, so no second request


def test_original_title_is_the_second_chance():
    original = ("track.getTopTags", "A", "Song (Remix)")
    session = FakeSession({original: tags_payload(("dance", 70))})
    assert client_for(session).tags_for("A", "Song (Remix)") == (["dance"], "track")


def test_plain_titles_are_only_asked_once():
    session = FakeSession({})
    client_for(session).tags_for("A", "Plain Title")
    assert [c for c in session.calls if c[0] == "track.getTopTags"] == [("track.getTopTags", "A", "Plain Title")]


# ---------------------------------------------------------------------------
# --diagnose: say WHY a lookup found nothing
# ---------------------------------------------------------------------------
KEY = ("track.getTopTags", "A", "B")


def verdict_for(payload, status=200):
    session = FakeSession({KEY: payload}, statuses={KEY: status})
    return client_for(session).diagnose_track("A", "B")


def test_diagnose_ok():
    verdict, detail = verdict_for(tags_payload(("rock", 100), ("dark", 40)))
    assert verdict == "OK" and "2 of 2" in detail


def test_diagnose_song_not_found():
    verdict, detail = verdict_for({"error": 6, "message": "Track not found"})
    assert verdict == "MISS" and "error 6" in detail


def test_diagnose_tags_all_below_threshold():
    verdict, detail = verdict_for(tags_payload(("rock", 7), ("pop", 3)))
    assert verdict == "FILTERED" and "highest is 7" in detail


def test_diagnose_song_exists_without_tags():
    assert verdict_for({"toptags": {"tag": []}})[0] == "MISS"


@pytest.mark.parametrize("payload, status", [
    ({"error": 29, "message": "Rate limit exceeded"}, 200),
    ({"error": 29, "message": "Rate limit exceeded"}, 429),
    ({"error": 10, "message": "Invalid API key"}, 403),
    (None, 503),
])
def test_diagnose_reports_errors_instead_of_hiding_them(payload, status):
    assert verdict_for(payload, status)[0] == "ERROR"


def test_diagnose_prints_a_line_per_song_and_a_summary(tracks_csv):
    lines = []
    session = FakeSession({
        ("track.getTopTags", "Artist A", "Popular"): tags_payload(("rock", 90)),
        ("track.getTopTags", "Artist C", "Middling"): {"error": 6, "message": "Track not found"},
    })
    tally = diagnose(load_targets(tracks_csv, min_popularity=50), client_for(session), 5, log=lines.append)
    assert tally == {"OK": 1, "MISS": 1}
    assert any("Artist A - Popular" in line for line in lines) and "Summary" in lines[-1]


# ---------------------------------------------------------------------------
# Temporary Last.fm errors are retried; "not found" is not
# ---------------------------------------------------------------------------
class FlakySession:
    """Fails with a given payload N times, then answers properly."""

    def __init__(self, failures, failing_payload, good_payload):
        self.failures, self.failing, self.good, self.calls = failures, failing_payload, good_payload, 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return FakeResponse(self.failing if self.calls <= self.failures else self.good)


def test_rate_limit_error_with_http_200_is_retried(monkeypatch):
    monkeypatch.setattr("fetch_tags.time.sleep", lambda s: None)
    session = FlakySession(2, {"error": 29, "message": "Rate limit"}, tags_payload(("rock", 90)))
    assert client_for(session).track_tags("A", "B") == ["rock"]
    assert session.calls == 3


def test_not_found_is_not_retried():
    session = FlakySession(99, {"error": 6, "message": "Track not found"}, None)
    assert client_for(session).track_tags("A", "B") == []
    assert session.calls == 1


def test_diagnose_samples_across_the_list_not_just_the_top(tracks_csv):
    lines = []
    diagnose(load_targets(tracks_csv, min_popularity=0), client_for(FakeSession({})), 3, log=lines.append)
    shown = " ".join(lines)
    assert "Unknown" in shown or "Middling" in shown  # not only the single most popular song
