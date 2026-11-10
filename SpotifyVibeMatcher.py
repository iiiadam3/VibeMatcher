import hashlib
import importlib
import pathlib
import random

import pandas as pd
import streamlit as st

# --- 1. עיצוב וקונפיגורציה ---
st.set_page_config(page_title="VibeMatcher Pro", layout="wide")
st.markdown("""
<style>
    .stApp { direction: rtl; }
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] ol,
    div[data-testid="stMarkdownContainer"] ul,
    h1, h2, h3 { text-align: right; }
    /* in RTL the list marker sits on the right, so the indent has to move with it */
    div[data-testid="stMarkdownContainer"] ol,
    div[data-testid="stMarkdownContainer"] ul { padding-right: 1.6rem; padding-left: 0; }
</style>
""", unsafe_allow_html=True)


# --- 2. Data + recommender live in recommender.py (testable without Streamlit) ---
#
# Streamlit Cloud usually deploys by re-running this file without restarting the
# process ("Pulling code changes... Updated app!" with no "Stopping..."). Python
# keeps an imported module in memory for the life of the process, so this file
# would be the new version while recommender stayed the old one, and the app died
# on a name or an argument that only the new version has.
#
# Reloading it explicitly, and keying the cached object on the file's contents,
# makes a deploy pick up both halves together.
import recommender as _recommender

_RECOMMENDER_VERSION = hashlib.sha256(
    pathlib.Path(_recommender.__file__).read_bytes()).hexdigest()[:12]
importlib.reload(_recommender)

from recommender import (  # noqa: E402  (must follow the reload above)
    MusicRecommender, artist_names, explain, load_data as _load_data, profile, song_label,
)


@st.cache_resource  # one shared copy for all users (cache_data would copy 575k rows on every click)
def get_recommender(version):
    """`version` is unused inside, but changing it makes Streamlit build a fresh
    recommender instead of handing back one built from the previous deploy's code."""
    return MusicRecommender(_load_data())


GENRE_MODE_LABELS = {
    "soft": "גמיש - עדיפות לז'אנר, אבל הוייב קובע",
    "strict": "קשוח - רק מהז'אנר שבחרתי",
    "off": "בלי ז'אנר - רק וייב",
}


RANDOM_OPTION = "🎲 Random (הפתיעו אותי)"

# How well known the recommendations must be. Obscure tracks hugely outnumber famous
# ones, so without a floor the closest matches are almost always unknown songs.
POPULARITY_LEVELS = {
    "להיטים בלבד": 65,
    "מוכרים (מומלץ)": 50,
    "כולל פחות מוכרים": 35,
    "הכל, גם שירים נדירים": 0,
}


def show_profile(row):
    """Bars that describe the song's sound."""
    for label, value, text in profile(row):
        c1, c2 = st.columns([1, 3])
        c1.write(f"**{label}**")
        c2.progress(value, text=f"\u2066{text}\u2069")  # keep "-7 dB" left-to-right inside the Hebrew page


def genre_text(row):
    return ", ".join(g for g in row["genres_list"] if g != "Other") or "לא ידוע"


# --- 3. שכבת הממשק (UI / Presentation) ---
def main():
    recommender = get_recommender(_RECOMMENDER_VERSION)
    df = recommender.df
    if df.empty:
        st.error("שגיאה בטעינת הנתונים.")
        return

    all_genres = df['genres_list'].explode().value_counts()
    popular_genres = sorted(g for g in all_genres[all_genres > 100].index if g != "Other")
    options_list = [RANDOM_OPTION] + popular_genres

    st.title("🎵 VibeMatcher")
    st.markdown("""
    **גלו מוזיקה לפי הוייב - לא רק לפי הז'אנר 💽** 
    המערכת משווה שירים לפי איך שהם *נשמעים*: קצב, אנרגיה, עוצמה ומצב רוח. 
    כך שיר ראפ יכול להוביל לשיר מטאל שמרגיש אותו דבר.

    **איך זה עובד?** 👇

    1. כתבו שיר שאתם אוהבים (שם השיר, ועדיף גם האמן)
    2. בחרו באיזה ז'אנר תרצו המלצות
    3. קבלו חמישה שירים עם אותו וייב, ולכל אחד הסבר למה הוא נבחר 🎶
    """)
    st.markdown("---")

    st.session_state.setdefault("song", "Do I Wanna Know Arctic Monkeys")
    col1, col2, col3 = st.columns(3)
    with col1:
        song_input = st.text_input("איזה שיר אהבת?", key="song",
                                   placeholder="לדוגמה: Do I Wanna Know Arctic Monkeys")
    with col2:
        selected_option = st.selectbox("באיזה ז'אנר תרצה המלצות?", options_list, key="genre")
    with col3:
        genre_mode = st.radio("כמה להקפיד על הז'אנר?", list(GENRE_MODE_LABELS),
                              format_func=GENRE_MODE_LABELS.get, key="mode")
    popularity_label = st.columns(3)[0].selectbox(
        "כמה מוכרים השירים שתקבלו?", list(POPULARITY_LEVELS), index=1, key="popularity")
    min_popularity = POPULARITY_LEVELS[popularity_label]
    st.caption("ℹ️ תגיות הז'אנר במאגר שייכות לאמן ולא לשיר, ולכן הן משוערות.")

    if not st.button("מצא לי מוזיקה", type="primary"):
        return

    if not song_input.strip():
        st.warning("כתבו קודם שם של שיר 🙂 (לדוגמה: Do I Wanna Know Arctic Monkeys)")
        return

    actual_genre = random.choice(popular_genres) if selected_option == RANDOM_OPTION else selected_option
    if selected_option == RANDOM_OPTION:
        st.info(f"🎲 נבחר עבורך הז'אנר: **{actual_genre}**")

    with st.spinner('מנתח את ה-Vibe...'):
        matches, source = recommender.recommend(
            song_input, actual_genre, genre_mode, min_popularity=min_popularity)

    if matches is None:
        st.error(source)
        return

    st.success(f"מצאתי: **{song_label(source)}**")
    st.caption(f"ז'אנר: {genre_text(source)}. לא השיר שחיפשת? הוסיפו את שם האמן לחיפוש.")
    with st.expander("📊 הפרופיל של השיר שבחרת", expanded=True):
        show_profile(source)

    st.subheader("אולי תאהב את אלו:")
    for _, row in matches.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"### {row['name']}")
                st.write(f"👤 {artist_names(row['artists'])}")
                st.write(f"🏷️ ז'אנר: {genre_text(row)}")
                if not row["in_target_genre"]:
                    st.caption("🔀 מחוץ לז'אנר שבחרת, נבחר בגלל הוייב")
            with c2:
                embed_code = f'<iframe src="https://open.spotify.com/embed/track/{row["id"]}" width="100%" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>'
                st.iframe(embed_code, height=90)
                st.caption("למה? " + " · ".join(explain(source, row)))


if __name__ == "__main__":
    main()
