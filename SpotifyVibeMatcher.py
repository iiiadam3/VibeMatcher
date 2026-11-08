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
from recommender import MusicRecommender, load_data as _load_data


@st.cache_resource  # one shared copy for all users (cache_data would copy 575k rows on every click)
def get_recommender():
    return MusicRecommender(_load_data())


GENRE_MODE_LABELS = {
    "soft": "גמיש - עדיפות לז'אנר, אבל הוייב קובע",
    "strict": "קשוח - רק מהז'אנר שבחרתי",
    "off": "בלי ז'אנר - רק וייב",
}


# --- 3. שכבת הממשק (UI / Presentation) ---
def main():
    recommender = get_recommender()
    df = recommender.df
    if df.empty:
        st.error("שגיאה בטעינת הנתונים.")
        return

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

    col1, col2, col3 = st.columns(3)
    with col1:
        song_input = st.text_input("איזה שיר אהבת?", "Do I Wanna Know Arctic Monkeys")
    with col2:
        selected_option = st.selectbox("באיזה ז'אנר תרצה המלצות?", options_list)
    with col3:
        genre_mode = st.radio("כמה להקפיד על הז'אנר?", list(GENRE_MODE_LABELS),
                              format_func=GENRE_MODE_LABELS.get)
    st.caption("ℹ️ תגיות הז'אנר במאגר שייכות לאמן ולא לשיר, ולכן הן משוערות.")

    if st.button("מצא לי מוזיקה"):
        actual_genre = random.choice(popular_genres) if "Random" in selected_option else selected_option
        if "Random" in selected_option:
            st.info(f"🎲 נבחר עבורך הז'אנר: **{actual_genre}**")

        with st.spinner('מנתח את ה-Vibe...'):
            matches, original_name = recommender.recommend(song_input, actual_genre, genre_mode)

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
                            if not row["in_target_genre"]:
                                st.caption(f"🔀 מחוץ לז'אנר ({', '.join(row['genres_list'])}), נבחר בגלל הוייב")
                        with c2:
                            track_id = row['id']
                            embed_code = f'<iframe src="https://open.spotify.com/embed/track/{track_id}" width="100%" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>'
                            st.iframe(embed_code, height=90)



if __name__ == "__main__":
    main()