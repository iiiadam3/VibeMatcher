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
from recommender import FEATURES, MusicRecommender, main_genres, norm_title  # noqa: E402


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
def make_song(name, artist, genre, popularity=50, **features):
    base = {"tempo": 120.0, "energy": 0.5, "loudness": -8.0, "danceability": 0.5,
            "valence": 0.5, "acousticness": 0.5, "instrumentalness": 0.0, "speechiness": 0.1}
    base.update(features)
    from recommender import norm_title as _n
    row = {"id": name, "name": name, "artists": f"['{artist}']", "genres_list": [genre],
           "popularity": popularity, "search_col": f"{name} {artist}".lower(),
           "norm_name": _n(name), **base}
    return row


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
    assert np.allclose(r.Z.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(r.Z.std(axis=0)[r.Z.std(axis=0) > 0], 1, atol=1e-5)


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


def test_rank_of_puts_nearest_song_first(catalog):
    r = MusicRecommender(catalog)
    assert r.rank_of("fast heavy rap", "fast heavy punk") == 1
    assert r.rank_of("fast heavy rap", "slow soft folk") > 1


# ---------------------------------------------------------------------------
# Empty search and explanations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query", ["", "   ", None])
def test_empty_search_returns_message_not_random_song(catalog, query):
    res, msg = MusicRecommender(catalog).recommend(query)
    assert res is None and "כתבו" in msg


def test_recommend_returns_source_row(catalog):
    _, source = MusicRecommender(catalog).recommend("medium pop", genre_mode="off")
    assert source["name"] == "medium pop"


def test_artist_names_and_label():
    from recommender import artist_names, song_label
    assert artist_names("['Justin Bieber', 'Giveon']") == "Justin Bieber, Giveon"
    row = {"name": "Do I Wanna Know?", "artists": "['Arctic Monkeys']", "release_date": "2013-06-18"}
    assert song_label(row) == "Arctic Monkeys – Do I Wanna Know? (2013)"


def test_explain_compares_tempo_and_mood():
    from recommender import explain
    src = {"tempo": 85, "energy": 0.5, "loudness": -8, "valence": 0.6}
    match = {"tempo": 86, "energy": 0.9, "loudness": -8.5, "valence": 0.2}
    reasons = explain(src, match)
    assert reasons[0] == "קצב דומה (86 מול 85 BPM)"
    assert "יותר אנרגטי" in reasons and "אפל יותר" in reasons


def test_profile_bars_stay_between_0_and_1():
    from recommender import profile
    extreme = {"tempo": 300, "energy": 1.0, "loudness": -60, "valence": 0.0, "danceability": 0.5}
    assert all(0.0 <= v <= 1.0 for _, v, _ in profile(extreme))


# ---------------------------------------------------------------------------
# Same song spelled two ways (curly vs straight apostrophe)
# ---------------------------------------------------------------------------
def test_norm_title_matches_curly_and_straight_apostrophes():
    assert norm_title("What\u2019s Next") == norm_title("What's Next")


def test_norm_title_ignores_case_and_punctuation():
    assert norm_title("Hello!") == norm_title("hello")


def test_norm_title_keeps_different_songs_apart():
    assert norm_title("Smells Like Teen Spirit") != norm_title("Smells Like Nirvana")


def test_the_searched_song_is_not_recommended_back_in_another_spelling(catalog):
    twin = make_song("medium pop", "E", "Pop", tempo=118, energy=0.6, loudness=-7, acousticness=0.3)
    twin["name"] = "Medium\u2019 Pop!"
    twin["id"] = "twin"
    twin["norm_name"] = norm_title(twin["name"])
    df = pd.concat([catalog, pd.DataFrame([twin])], ignore_index=True)
    res, _ = MusicRecommender(df).recommend("medium pop", genre_mode="off", n=5)
    assert "Medium\u2019 Pop!" not in res["name"].tolist()


# ---------------------------------------------------------------------------
# Popularity floor
# ---------------------------------------------------------------------------
@pytest.fixture
def mixed_popularity():
    return pd.DataFrame([
        make_song("the source", "A", "Rap", popularity=80, tempo=150, energy=0.8),
        make_song("obscure twin", "B", "Rap", popularity=10, tempo=150, energy=0.8),
        make_song("famous but further", "C", "Rap", popularity=90, tempo=120, energy=0.5),
    ])


def test_popularity_floor_excludes_obscure_songs(mixed_popularity):
    res, _ = MusicRecommender(mixed_popularity).recommend(
        "the source", genre_mode="off", n=5, min_popularity=50)
    assert res["name"].tolist() == ["famous but further"]


def test_without_a_floor_the_closest_song_wins(mixed_popularity):
    res, _ = MusicRecommender(mixed_popularity).recommend("the source", genre_mode="off", n=5)
    assert res.iloc[0]["name"] == "obscure twin"


def test_floor_that_excludes_everything_returns_a_helpful_message(mixed_popularity):
    res, msg = MusicRecommender(mixed_popularity).recommend(
        "the source", genre_mode="off", min_popularity=99)
    assert res is None and "פופולריות" in msg
