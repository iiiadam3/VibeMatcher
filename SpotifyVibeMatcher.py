import streamlit as st
import pandas as pd
import random

# --- 1. עיצוב וקונפיגורציה ---
st.set_page_config(page_title="VibeMatcher Pro", layout="wide")
st.markdown("""
<style>
    .stApp { direction: rtl; }
    div[data-testid="stMarkdownContainer"] p, h1, h2, h3 { text-align: right; }
</style>
""", unsafe_allow_html=True)


# --- 2. Data + recommender live in recommender.py (testable without Streamlit) ---
from recommender import MusicRecommender, artist_names, explain, load_data as _load_data, profile, song_label


@st.cache_resource  # one shared copy for all users (cache_data would copy 575k rows on every click)
def get_recommender():
    return MusicRecommender(_load_data())


GENRE_MODE_LABELS = {
    "soft": "גמיש - עדיפות לז'אנר, אבל הוייב קובע",
    "strict": "קשוח - רק מהז'אנר שבחרתי",
    "off": "בלי ז'אנר - רק וייב",
}


RANDOM_OPTION = "🎲 Random (הפתיעו אותי)"
EXAMPLE = {"song": "Magnolia Playboi Carti", "genre": "Metal", "mode": "soft"}


def load_example():
    """Runs when the example button is clicked, before the widgets are drawn again."""
    st.session_state["song"] = EXAMPLE["song"]
    st.session_state["genre"] = EXAMPLE["genre"]
    st.session_state["mode"] = EXAMPLE["mode"]
    st.session_state["auto_search"] = True


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
    recommender = get_recommender()
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
    st.button(f"✨ נסו דוגמה: Playboi Carti – Magnolia ← {EXAMPLE['genre']}", on_click=load_example)
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
    st.caption("ℹ️ תגיות הז'אנר במאגר שייכות לאמן ולא לשיר, ולכן הן משוערות.")

    clicked = st.button("מצא לי מוזיקה", type="primary")
    if not (clicked or st.session_state.pop("auto_search", False)):
        return

    if not song_input.strip():
        st.warning("כתבו קודם שם של שיר 🙂 (לדוגמה: Do I Wanna Know Arctic Monkeys)")
        return

    actual_genre = random.choice(popular_genres) if selected_option == RANDOM_OPTION else selected_option
    if selected_option == RANDOM_OPTION:
        st.info(f"🎲 נבחר עבורך הז'אנר: **{actual_genre}**")

    with st.spinner('מנתח את ה-Vibe...'):
        matches, source = recommender.recommend(song_input, actual_genre, genre_mode)

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
