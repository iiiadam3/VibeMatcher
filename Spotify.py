import streamlit as st
import pandas as pd
import numpy as np
import ast
import random

# --- 1. עיצוב וקונפיגורציה ---
st.set_page_config(page_title="VibeMatcher Pro", layout="wide")
st.markdown("""
<style>
    .stApp { direction: rtl; }
    div[data-testid="stMarkdownContainer"] p, h1, h2, h3 { text-align: right; }
</style>
""", unsafe_allow_html=True)


# --- 2. מודול עיבוד נתונים (Data Pipeline) ---
# --- 2. מודול עיבוד נתונים (Data Pipeline) ---
@st.cache_data
def load_data():
    tracks = pd.read_csv('tracks.zip')
    artists = pd.read_csv('artists.csv')

    tracks['artist_id'] = tracks['id_artists'].str.replace(r"[\[\]\']", "", regex=True).str.split(',').str[
        0].str.strip()
    df = pd.merge(tracks, artists[['id', 'genres']], left_on='artist_id', right_on='id', how='left')
    df = df.dropna(subset=['name', 'artists', 'genres'])

    def simplify_genres(genre_str):
        try:
            genres_list = ast.literal_eval(genre_str)
        except (ValueError, SyntaxError):
            return ['Other']

        simplified = set()

        # 1. רשימת "מילות אזהרה" לזיהוי שפות זרות
        foreign_prefixes = [
            'french', 'german', 'italian', 'czech', 'polish', 'russian',
            'turkish', 'swedish', 'spanish', 'dutch', 'finnish', 'danish',
            'norwegian', 'portuguese', 'greek'
        ]

        # 2. המילון הרגיל והנקי שלנו
        genre_mapping = {
            'uk drill': 'UK Rap', 'uk hip hop': 'UK Rap', 'grime': 'UK Rap',
            'israeli rap': 'Israeli Rap', 'israeli rock': 'Israeli Rock', 'israeli': 'Israeli (General)',
            'korean rap': 'Korean Rap', 'k-pop': 'K-Pop',
            'alternative rock': 'Alternative Rock', 'hard rock': 'Hard Rock',
            'punk': 'Punk / Punk Rock', 'progressive rock': 'Progressive Rock', 'pop rock': 'Pop-Rock',
            'thrash metal': 'Thrash Metal', 'death metal': 'Death Metal', 'heavy metal': 'Heavy Metal',
            'indie pop': 'Indie Pop', 'pop rap': 'Pop', 'hip pop': 'Pop', 'dance pop': 'Pop',
            'trip hop': 'Trip Hop', 'r&b': 'R&B / Soul', 'soul': 'R&B / Soul',
            'urban contemporary': 'R&B / Soul', 'new jack swing': 'R&B / Soul',
            'latin': 'Latin Music', 'country': 'Country Music', 'reggae': 'Reggae', 'dancehall': 'Dancehall',
            'drill': 'Rap', 'trap': 'Rap', 'hip hop': 'Rap', 'rap': 'Rap',
            'rock': 'Rock', 'metal': 'Metal', 'pop': 'Pop', 'jazz': 'Jazz',
            'electro': 'Electronic/Dance', 'techno': 'Electronic/Dance',
            'house': 'Electronic/Dance', 'edm': 'Electronic/Dance'
        }

        for g in genres_list:
            g_lower = g.lower()

            # --- פילטר מקדים: זיהוי ראפ בינלאומי חכם ---
            if 'rap' in g_lower or 'hip hop' in g_lower or 'trap' in g_lower:
                if any(foreign_word in g_lower for foreign_word in foreign_prefixes):
                    simplified.add('International Rap')
                    continue  # מצאנו שפה זרה? מדלגים הלאה כדי שזה לא ייפול לראפ הרגיל

            # --- חיפוש רגיל במילון ---
            for key, display_name in genre_mapping.items():
                if key in g_lower:
                    simplified.add(display_name)
                    break

        return list(simplified) if simplified else ['Other']

    df['genres_list'] = df['genres'].apply(simplify_genres)
    df['search_col'] = df['name'].str.lower() + " " + df['artists'].str.lower()

    if 'tempo' in df.columns:
        df['tempo_norm'] = df['tempo'] / df['tempo'].max()

    return df

# --- 3. מנוע ההמלצות (Core Logic) ---
class MusicRecommender:
    def __init__(self, dataframe):
        self.df = dataframe
        self.features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'speechiness',
                         'tempo_norm']
        self.weights = np.array([1.0, 1.5, 1.0, 3.0, 2.0, 2.5, 1.5])

    def recommend(self, source_query, target_genre, n=5):
        words = source_query.lower().split()
        mask = np.ones(len(self.df), dtype=bool)
        for word in words:
            mask &= self.df['search_col'].str.contains(word, na=False)
        source_results = self.df[mask]

        if source_results.empty:
            return None, f"לא מצאתי את השיר '{source_query}' במאגר."

        source_row = source_results.iloc[0]
        source_vec = source_row[self.features].values.astype(float)

        target_df = self.df[self.df['genres_list'].apply(lambda g: target_genre in g)].copy()
        target_df = target_df[target_df['name'] != source_row['name']]

        if target_df.empty:
            return None, f"לא מצאתי מספיק שירים בז'אנר {target_genre}."

        matrix = target_df[self.features].values.astype(float)
        diff = (matrix - source_vec) * self.weights
        distances = np.sqrt(np.sum(diff ** 2, axis=1))
        target_df['dist'] = distances

        return target_df.sort_values('dist').drop_duplicates(subset=['name', 'artists']).head(n), source_row['name']


# --- 4. שכבת הממשק (UI / Presentation) ---
def main():
    df = load_data()
    if df.empty:
        st.error("שגיאה בטעינת הנתונים.")
        return

    recommender = MusicRecommender(df)
    all_genres = df['genres_list'].explode().value_counts()
    popular_genres = sorted(all_genres[all_genres > 100].index.tolist())
    options_list = ["🎲 Random (הפתיעו אותי)"] + popular_genres

    st.title("🎵 VibeMatcher")
    st.markdown("""
    **גלו מוזיקה לפי הוייב - לא רק לפי הז'אנר 💽** 
    המערכת מנתחת את ה-DNA האקוסטי של השירים ומוצאת דמיון מתמטי בין מאפייני קצב, אנרגיה ודינמיקה. 
    התוצאה היא חיבורים מפתיעים שחוצים הגדרות וז'אנרים.

    **איך זה עובד?** 👇

    פשוט מזינים שיר שאתם אוהבים, בוחרים ז'אנר (או נותנים למערכת להפתיע אתכם), ובלחיצת כפתור תקבלו חמישה שירים חדשים שחולקים את אותה ההרגשה 🎶
    """)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        song_input = st.text_input("איזה שיר אהבת?", "Do I Wanna Know Arctic Monkeys")
    with col2:
        selected_option = st.selectbox("באיזה ז'אנר תרצה המלצות?", options_list)

    if st.button("מצא לי מוזיקה"):
        actual_genre = random.choice(popular_genres) if "Random" in selected_option else selected_option
        if "Random" in selected_option:
            st.info(f"🎲 נבחר עבורך הז'אנר: **{actual_genre}**")

        with st.spinner('מנתח את ה-Vibe...'):
            matches, original_name = recommender.recommend(song_input, actual_genre)

            if matches is None:
                st.error(original_name)
            else:
                st.success(f"מבוסס על **{original_name}**, אולי תאהב את אלו:")
                for _, row in matches.iterrows():
                    with st.container():
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.markdown(f"### {row['name']}")
                            st.write(f"👤 {row['artists']}")
                        with c2:
                            track_id = row['id_x']
                            embed_code = f'<iframe src="https://open.spotify.com/embed/track/{track_id}" width="100%" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>'
                            st.iframe(embed_code, height=90)


# נקודת הכניסה של התוכנית (Entry Point)
if __name__ == "__main__":
    main()