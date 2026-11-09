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
    """Load only the columns the app uses, to stay inside Streamlit Cloud's memory limit."""
    tracks = pd.read_csv(
        tracks_path,
        usecols=["id", "name", "artists", "id_artists", "popularity", "release_date", *FEATURES],
    )
    artists = pd.read_csv(artists_path, usecols=["id", "genres"])

    # Genres in this dataset belong to ARTISTS, not songs. We use the first artist.
    tracks["artist_id"] = (
        tracks["id_artists"].str.replace(r"[\[\]\']", "", regex=True).str.split(",").str[0].str.strip()
    )
    df = tracks.drop(columns="id_artists").merge(
        artists.rename(columns={"id": "artist_id"}), on="artist_id", how="left"
    )
    del tracks, artists
    df = df.dropna(subset=["name", "artists", "genres"])
    df = df[df["tempo"] > 0]  # tempo 0 = broken measurement

    # Many songs share an artist, so parse each distinct genre string only once.
    unique_genres = df["genres"].unique()
    parsed = dict(zip(unique_genres, map(main_genres, unique_genres)))
    df["genres_list"] = df["genres"].map(parsed)
    df["search_col"] = df["name"].str.lower() + " " + df["artists"].str.lower()
    df[FEATURES] = df[FEATURES].astype("float32")
    return df.drop(columns=["genres", "artist_id"]).reset_index(drop=True)


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
        X = self.df[FEATURES].to_numpy(dtype="float64")
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-12] = 1.0  # a feature that never changes can't separate songs; avoid dividing by 0
        self.Z = ((X - self.mean) / self.std).astype("float32")
        self._genre_cache = {}

    def find_song(self, query):
        """Return the row index of the best match for a search, or None.
        If several versions match, take the most popular one (usually the original)."""
        if not query or not query.strip():
            return None  # an empty search would match every song
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

    def in_genre(self, genre):
        """True/False per song: is `genre` one of its main genres? Cached per genre."""
        if genre not in self._genre_cache:
            self._genre_cache[genre] = self.df["genres_list"].map(lambda g: genre in g).to_numpy()
        return self._genre_cache[genre]

    def _scored(self, src, target_genre, genre_mode):
        """Distance to every song after applying the genre mode (inf = excluded)."""
        if genre_mode not in GENRE_MODES:
            raise ValueError(f"genre_mode must be one of {GENRE_MODES}")
        dist = self.distances(src)
        if target_genre and genre_mode != "off":
            in_g = self.in_genre(target_genre)
            if genre_mode == "strict":
                dist = np.where(in_g, dist, np.inf)
            else:
                dist = dist + np.where(in_g, 0.0, SOFT_GENRE_PENALTY)
        # Drop the source song itself (all its versions).
        same = (self.df["artists"].to_numpy() == self.df.at[src, "artists"]) & (
            self.df["name"].str.lower().to_numpy() == self.df.at[src, "name"].lower())
        return np.where(same, np.inf, dist)

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
        if not query or not query.strip():
            return None, "כתבו שם של שיר (ואפשר גם אמן) כדי שנדע ממה להתחיל."
        src = self.find_song(query)
        if src is None:
            return None, f"לא מצאתי את השיר '{query}' במאגר."

        dist = self._scored(src, target_genre, genre_mode)
        # Only build a table for the closest few hundred songs, not all 575k.
        k = min(len(dist), max(n * 50, 200))
        top = np.argpartition(dist, k - 1)[:k]
        top = top[np.isfinite(dist[top])]
        result = self.df.iloc[top].assign(dist=dist[top])
        result["in_target_genre"] = self.in_genre(target_genre)[top] if target_genre else True
        result = result.sort_values("dist").drop_duplicates(subset=["name", "artists"]).head(n)

        if result.empty:
            return None, f"לא מצאתי מספיק שירים בז'אנר {target_genre}."
        return result, self.df.loc[src]

    def rank_of(self, query_a, query_b, target_genre=None, genre_mode="off"):
        """Where does song B land in song A's results? (1 = top). Used by the golden-pair tests."""
        a, b = self.find_song(query_a), self.find_song(query_b)
        if a is None or b is None:
            return None
        dist = self._scored(a, target_genre, genre_mode)
        # Song B may exist in several versions; use whichever version is closest.
        same_b = ((self.df["name"] == self.df.at[b, "name"]) & (self.df["artists"] == self.df.at[b, "artists"])).to_numpy()
        best_b = dist[same_b].min()
        if not np.isfinite(best_b):
            return None
        closer = self.df.loc[dist < best_b, ["name", "artists"]].drop_duplicates()
        return len(closer) + 1


# ---------------------------------------------------------------------------
# 5. Explaining results to the user
# ---------------------------------------------------------------------------
def artist_names(artists_str):
    """"['Justin Bieber', 'Giveon']" -> "Justin Bieber, Giveon"."""
    try:
        return ", ".join(ast.literal_eval(artists_str))
    except (ValueError, SyntaxError):
        return str(artists_str)


def song_label(row):
    """'Arctic Monkeys – Do I Wanna Know? (2013)'"""
    year = str(row.get("release_date", ""))[:4]
    return f"{artist_names(row['artists'])} – {row['name']}" + (f" ({year})" if year.isdigit() else "")


def _compare(diff, close, same, more, less):
    if abs(diff) <= close:
        return same
    return more if diff > 0 else less


def explain(source, match):
    """Short Hebrew reasons comparing a match to the source song, e.g. 'קצב דומה (86 מול 85 BPM)'."""
    t_s, t_m = round(source["tempo"]), round(match["tempo"])
    return [
        _compare(t_m - t_s, 8, f"קצב דומה ({t_m} מול {t_s} BPM)",
                 f"מהיר יותר ({t_m} מול {t_s} BPM)", f"איטי יותר ({t_m} מול {t_s} BPM)"),
        _compare(match["energy"] - source["energy"], 0.1, "אנרגיה דומה", "יותר אנרגטי", "רגוע יותר"),
        _compare(match["loudness"] - source["loudness"], 2.0, "עוצמה דומה", "חזק יותר", "שקט יותר"),
        _compare(match["valence"] - source["valence"], 0.12, "מצב רוח דומה", "שמח יותר", "אפל יותר"),
    ]


def _scale(value, low, high):
    return float(min(max((value - low) / (high - low), 0.0), 1.0))


def profile(row):
    """Values for the song-profile bars: (label, 0-1 bar value, text shown next to it)."""
    return [
        ("קצב", _scale(row["tempo"], 60, 200), f"{round(row['tempo'])} BPM"),
        ("אנרגיה", float(row["energy"]), f"{row['energy']:.0%}"),
        ("עוצמה", _scale(row["loudness"], -26, -2), f"{row['loudness']:.0f} dB"),
        ("מצב רוח (אפל ← שמח)", float(row["valence"]), f"{row['valence']:.0%}"),
        ("רקידות", float(row["danceability"]), f"{row['danceability']:.0%}"),
    ]
