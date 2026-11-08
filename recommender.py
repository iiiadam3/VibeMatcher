"""
VibeMatcher core logic: data loading, genre cleanup, and the recommender.

Kept separate from the Streamlit UI so it can be tested with pytest.
"""
import ast
from collections import Counter

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Which audio features describe a "vibe", and how much each one matters.
#
# Every feature is standardized first (see MusicRecommender), so a weight of
# 2.0 really means "twice as important as a weight of 1.0".
# ---------------------------------------------------------------------------
FEATURES = [
    "tempo",             # how fast
    "energy",            # how intense
    "loudness",          # how heavy / loud the mix is
    "danceability",      # how driving the rhythm is
    "valence",           # dark (low) vs bright (high)
    "acousticness",      # acoustic vs produced
    "instrumentalness",  # vocals vs no vocals
    "speechiness",       # talky / rap delivery - kept low so it doesn't lock results into rap
]

DEFAULT_WEIGHTS = {
    "tempo": 2.0,
    "energy": 2.0,
    "loudness": 1.5,
    "danceability": 0.5,  # Spotify rates rap as more "danceable" than rock, so this acts like a hidden genre signal
    "valence": 1.0,
    "acousticness": 1.0,
    "instrumentalness": 0.5,
    "speechiness": 0.3,
}

# How strict the genre filter is (chosen by the user in the UI).
GENRE_MODES = ("off", "soft", "strict")

# In "soft" mode, a song outside the target genre gets this much extra distance.
# Distances are in "standardized" units, so 1.0 is roughly one feature being one
# spread away. Small enough that a great vibe match from another genre can still win.
SOFT_GENRE_PENALTY = 1.0


# ---------------------------------------------------------------------------
# 2. Genre cleanup
# ---------------------------------------------------------------------------
FOREIGN_PREFIXES = [
    "french", "german", "italian", "czech", "czsk", "slovak", "polish", "russian",
    "turkish", "swedish", "spanish", "dutch", "finnish", "danish", "norwegian",
    "portuguese", "greek", "icelandic", "hungarian", "romanian", "ukrainian",
    "japanese", "brazilian", "argentine", "mexican", "chilean", "belgian",
    "austrian", "swiss", "croatian", "serbian", "latvian", "lithuanian",
    "estonian", "indonesian", "thai", "vietnamese", "arab", "kazakh",
    "espanol", "francais", "deutsch", "kreyol", "taiwan", "chinese", "cantonese", "mandarin",
]

# Order matters: the first key found inside a tag wins.
GENRE_MAPPING = {
    "uk drill": "UK Rap", "uk hip hop": "UK Rap", "grime": "UK Rap",
    "israeli rap": "Israeli Rap", "israeli rock": "Israeli Rock", "israeli": "Israeli (General)",
    "korean rap": "Korean Rap", "k-pop": "K-Pop",
    "alternative rock": "Alternative Rock", "hard rock": "Hard Rock",
    "punk": "Punk / Punk Rock", "progressive rock": "Progressive Rock", "pop rock": "Pop-Rock",
    "thrash metal": "Thrash Metal", "death metal": "Death Metal", "heavy metal": "Heavy Metal",
    "indie pop": "Indie Pop", "pop rap": "Pop", "hip pop": "Pop", "dance pop": "Pop",
    "trip hop": "Trip Hop", "r&b": "R&B / Soul", "soul": "R&B / Soul",
    "urban contemporary": "R&B / Soul", "new jack swing": "R&B / Soul",
    "latin": "Latin Music", "country": "Country Music", "reggae": "Reggae", "dancehall": "Dancehall",
    "drill": "Rap", "trap": "Rap", "hip hop": "Rap", "rap": "Rap",
    "rock": "Rock", "metal": "Metal", "pop": "Pop", "jazz": "Jazz",
    "electro": "Electronic/Dance", "techno": "Electronic/Dance",
    "house": "Electronic/Dance", "edm": "Electronic/Dance",
}


def map_tag(tag):
    """Map one raw Spotify artist tag (e.g. 'atl hip hop') to a simple genre, or None."""
    t = tag.lower()
    if any(w in t for w in ("rap", "hip hop", "trap")) and any(p in t for p in FOREIGN_PREFIXES):
        return "International Rap"
    for key, name in GENRE_MAPPING.items():
        if key in t:
            return name
    return None


def main_genres(genre_str):
    """
    Pick an artist's MAIN genre(s) by majority vote over their tags.

    Before: one matching tag was enough, so Brandy (one 'hip hop' tag among
    four R&B tags) counted as Rap. Now each tag is one vote and only the
    genre(s) with the most votes are kept (ties keep all winners).
    """
    try:
        tags = ast.literal_eval(genre_str)
    except (ValueError, SyntaxError):
        return ["Other"]

    votes = Counter(g for g in (map_tag(t) for t in tags) if g)
    # One foreign-rap tag is enough to know the artist raps in another language,
    # even if other tags (e.g. 'rap francais') slipped past the prefix list.
    if votes["International Rap"]:
        votes["International Rap"] += votes.pop("Rap", 0)
    if not votes:
        return ["Other"]
    top = max(votes.values())
    return sorted(g for g, v in votes.items() if v == top)


# ---------------------------------------------------------------------------
# 3. Data loading
# ---------------------------------------------------------------------------
def load_data(tracks_path="tracks.zip", artists_path="artists.zip"):
    tracks = pd.read_csv(tracks_path)
    artists = pd.read_csv(artists_path)

    # Genres in this dataset belong to ARTISTS, not songs. We use the first artist.
    tracks["artist_id"] = (
        tracks["id_artists"].str.replace(r"[\[\]\']", "", regex=True).str.split(",").str[0].str.strip()
    )
    df = tracks.merge(
        artists[["id", "genres"]].rename(columns={"id": "artist_key"}),
        left_on="artist_id", right_on="artist_key", how="left",
    )
    df = df.dropna(subset=["name", "artists", "genres"])
    df = df[df["tempo"] > 0]  # tempo 0 = broken measurement

    df["genres_list"] = df["genres"].apply(main_genres)
    df["search_col"] = df["name"].str.lower() + " " + df["artists"].str.lower()
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. The recommender
# ---------------------------------------------------------------------------
class MusicRecommender:
    def __init__(self, dataframe, weights=None):
        self.df = dataframe.reset_index(drop=True)
        w = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.weights = np.array([w[f] for f in FEATURES], dtype=float)

        # Standardize: (value - average) / spread, per feature.
        # Afterwards every feature has average 0 and spread 1, so tempo (50-220)
        # and energy (0-1) are on the same footing.
        X = self.df[FEATURES].to_numpy(dtype=float)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-12] = 1.0  # a feature that never changes can't separate songs; avoid dividing by 0
        self.Z = (X - self.mean) / self.std

    def find_song(self, query):
        """Return the row index of the best match for a search, or None.
        If several versions match, take the most popular one (usually the original)."""
        mask = np.ones(len(self.df), dtype=bool)
        for word in query.lower().split():
            mask &= self.df["search_col"].str.contains(word, regex=False, na=False).to_numpy()
        matches = self.df[mask]
        if matches.empty:
            return None
        return matches["popularity"].idxmax()

    def distances(self, source_idx):
        """Weighted Euclidean distance from one song to every song."""
        diff = (self.Z - self.Z[source_idx]) * self.weights
        return np.sqrt((diff ** 2).sum(axis=1))

    def recommend(self, query, target_genre=None, genre_mode="soft", n=5):
        """
        Returns (results_dataframe, source_song_name) or (None, error_message).

        genre_mode:
          "off"    - pure vibe, genre ignored
          "soft"   - songs outside target_genre get SOFT_GENRE_PENALTY extra distance
          "strict" - only songs in target_genre
        """
        if genre_mode not in GENRE_MODES:
            raise ValueError(f"genre_mode must be one of {GENRE_MODES}")

        src = self.find_song(query)
        if src is None:
            return None, f"לא מצאתי את השיר '{query}' במאגר."
        source = self.df.loc[src]

        dist = self.distances(src)
        result = self.df.assign(dist=dist)
        in_genre = result["genres_list"].apply(lambda g: target_genre in g)
        result = result.assign(in_target_genre=in_genre)

        if target_genre and genre_mode == "strict":
            result = result[in_genre]
        elif target_genre and genre_mode == "soft":
            result = result.assign(dist=result["dist"] + np.where(in_genre, 0.0, SOFT_GENRE_PENALTY))

        # Drop the source song itself (all its versions) from the results.
        result = result[~((result["name"].str.lower() == source["name"].lower())
                          & (result["artists"] == source["artists"]))]
        result = result.sort_values("dist").drop_duplicates(subset=["name", "artists"]).head(n)

        if result.empty:
            return None, f"לא מצאתי מספיק שירים בז'אנר {target_genre}."
        return result, source["name"]

    def rank_of(self, query_a, query_b, target_genre=None, genre_mode="off"):
        """Where does song B land in song A's results? (1 = top). Used by the golden-pair tests."""
        a, b = self.find_song(query_a), self.find_song(query_b)
        if a is None or b is None:
            return None
        res, _ = self.recommend(query_a, target_genre, genre_mode, n=len(self.df))
        b_row = self.df.loc[b]
        hits = np.flatnonzero((res["name"] == b_row["name"]).to_numpy()
                              & (res["artists"] == b_row["artists"]).to_numpy())
        return int(hits[0]) + 1 if len(hits) else None
