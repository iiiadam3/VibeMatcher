"""
Fast tests: no Streamlit and no big dataset needed.
They use a tiny hand-made catalog so every expected answer can be checked by eye.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recommender import FEATURES, MusicRecommender, main_genres  # noqa: E402


# ---------------------------------------------------------------------------
# Genre cleanup
# ---------------------------------------------------------------------------
def test_majority_vote_beats_single_tag():
    # Brandy's real tags: one hip hop tag shouldn't make her a rapper.
    tags = "['dance pop', 'hip hop', 'hip pop', 'neo soul', 'pop', 'pop r&b', 'pop rap', 'r&b', 'urban contemporary']"
    assert "Rap" not in main_genres(tags)
    assert "R&B / Soul" in main_genres(tags)


def test_pure_rapper_is_rap():
    assert main_genres("['atl hip hop', 'rap', 'trap']") == ["Rap"]


def test_foreign_rap_tag_wins_over_plain_rap():
    # 'rap francais' has no 'french' prefix, but 'french hip hop' tells us enough.
    assert main_genres("['french hip hop', 'rap francais']") == ["International Rap"]


def test_bad_genre_string_is_other():
    assert main_genres("not a list") == ["Other"]
    assert main_genres("[]") == ["Other"]


# ---------------------------------------------------------------------------
# Recommender on a tiny catalog
# ---------------------------------------------------------------------------
def make_song(name, artist, genre, **features):
    base = {"tempo": 120.0, "energy": 0.5, "loudness": -8.0, "danceability": 0.5,
            "valence": 0.5, "acousticness": 0.5, "instrumentalness": 0.0, "speechiness": 0.1}
    base.update(features)
    return {"id": name, "name": name, "artists": f"['{artist}']", "genres_list": [genre],
            "popularity": 50, "search_col": f"{name} {artist}".lower(), **base}


@pytest.fixture
def catalog():
    songs = [
        make_song("fast heavy rap", "A", "Rap", tempo=160, energy=0.9, loudness=-4, acousticness=0.0),
        make_song("fast heavy punk", "B", "Punk", tempo=158, energy=0.9, loudness=-4, acousticness=0.0),
        make_song("slow soft rap", "C", "Rap", tempo=70, energy=0.2, loudness=-15, acousticness=0.9),
        make_song("slow soft folk", "D", "Folk", tempo=72, energy=0.2, loudness=-15, acousticness=0.9),
        make_song("medium pop", "E", "Pop", tempo=118, energy=0.6, loudness=-7, acousticness=0.3),
        make_song("medium rap", "F", "Rap", tempo=115, energy=0.6, loudness=-7, acousticness=0.3),
    ]
    return pd.DataFrame(songs)


def test_features_are_standardized(catalog):
    r = MusicRecommender(catalog)
    # Every feature should now have average ~0 and spread ~1, so none dominates by its units.
    assert np.allclose(r.Z.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(r.Z.std(axis=0)[r.Z.std(axis=0) > 0], 1)


def test_vibe_crosses_genre_when_genre_off(catalog):
    res, _ = MusicRecommender(catalog).recommend("fast heavy rap", genre_mode="off", n=1)
    assert res.iloc[0]["name"] == "fast heavy punk"


def test_strict_mode_only_returns_target_genre(catalog):
    res, _ = MusicRecommender(catalog).recommend("fast heavy punk", "Rap", "strict", n=5)
    assert all("Rap" in g for g in res["genres_list"])


def test_soft_mode_prefers_genre_but_allows_others(catalog):
    res, _ = MusicRecommender(catalog).recommend("fast heavy rap", "Folk", "soft", n=5)
    assert res["in_target_genre"].any()
    assert not res["in_target_genre"].all()


def test_source_song_is_never_recommended(catalog):
    res, _ = MusicRecommender(catalog).recommend("medium pop", genre_mode="off", n=5)
    assert "medium pop" not in res["name"].tolist()


def test_unknown_song_returns_error_message(catalog):
    res, msg = MusicRecommender(catalog).recommend("song that does not exist")
    assert res is None and "לא מצאתי" in msg


def test_returns_n_results(catalog):
    res, _ = MusicRecommender(catalog).recommend("medium pop", genre_mode="off", n=3)
    assert len(res) == 3


def test_bad_genre_mode_raises(catalog):
    with pytest.raises(ValueError):
        MusicRecommender(catalog).recommend("medium pop", genre_mode="maybe")


def test_all_features_present(catalog):
    assert set(FEATURES) <= set(catalog.columns)
