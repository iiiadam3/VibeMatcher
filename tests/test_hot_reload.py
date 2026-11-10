"""
Guards against the deploy failure that took the live app down twice.

Streamlit Cloud usually deploys by re-running the main script without restarting
the process. Python keeps imported modules in memory for the life of the process,
so the main script became the new version while `recommender` stayed the old one:

    ImportError: cannot import name 'artist_names' from 'recommender'
    TypeError: recommend() got an unexpected keyword argument 'min_popularity'

Both times the code on disk was correct and the tests passed. These tests plant a
stale module in memory first, the way the running server had one, and then start
the app.
"""
import sys
import types
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "SpotifyVibeMatcher.py"
RECOMMENDER = Path(__file__).resolve().parents[1] / "recommender.py"


@pytest.fixture
def stale_recommender_in_memory():
    """Put a module named `recommender` in memory that is missing recent additions."""
    saved = sys.modules.get("recommender")

    stale = types.ModuleType("recommender")
    stale.__file__ = str(RECOMMENDER)

    class OutdatedRecommender:
        def __init__(self, dataframe):
            raise AssertionError("the stale module was used instead of the file on disk")

    stale.MusicRecommender = OutdatedRecommender
    # Deliberately missing: artist_names, explain, profile, song_label, load_data.
    sys.modules["recommender"] = stale
    yield
    if saved is not None:
        sys.modules["recommender"] = saved
    else:
        sys.modules.pop("recommender", None)


def test_app_starts_even_when_a_stale_module_is_in_memory(stale_recommender_in_memory):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=300).run()
    assert not app.exception, [str(e.value) for e in app.exception]


def test_a_search_works_after_a_hot_reload(stale_recommender_in_memory):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=300).run()
    app.text_input(key="song").input("Do I Wanna Know Arctic Monkeys")
    app.selectbox(key="genre").select("Rock")
    [b for b in app.button if b.label == "מצא לי מוזיקה"][0].click().run()

    assert not app.exception, [str(e.value) for e in app.exception]
    assert len([m for m in app.markdown if m.value.startswith("###")]) == 5


def test_the_cached_recommender_is_keyed_on_the_file_contents():
    """A deploy that changes recommender.py must not reuse the old cached object."""
    source = APP.read_text(encoding="utf-8")
    assert "hashlib.sha256(" in source and "read_bytes()" in source
    assert "get_recommender(_RECOMMENDER_VERSION)" in source


def test_the_module_is_reloaded_before_its_names_are_imported():
    source = APP.read_text(encoding="utf-8")
    reload_at = source.index("importlib.reload(_recommender)")
    import_at = source.index("from recommender import (")
    assert reload_at < import_at, "names must be imported after the reload, not before"
