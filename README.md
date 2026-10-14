# 🎵 VibeMatcher

VibeMatcher is a data-driven, cross-genre music recommendation engine built with Python. Instead of relying on traditional metadata (like genres or artists), it uses acoustic feature analysis to find songs with a mathematically similar "vibe" across completely different musical categories.

## 🚀 Features
* **Acoustic DNA Matching:** Utilizes a custom K-Nearest Neighbors (KNN) logic to calculate the weighted Euclidean distance between tracks based on 7 audio features (Danceability, Energy, Valence, Acousticness, Instrumentalness, Speechiness, and Tempo).
* **Smart Data Pipeline:** Efficiently loads and preprocesses a compressed dataset of over 600,000 Spotify tracks and artists using Pandas, including dynamic text cleaning and genre mapping.
* **Interactive UI:** Built with Streamlit, offering a smooth, localized (RTL) user experience with a "Random" discovery feature.
* **Spotify Integration:** Embeds interactive Spotify web players directly into the search results using dynamic iframes.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy
* **Frontend/UI:** Streamlit
* **Algorithms:** Vector Mathematics, K-Nearest Neighbors (Custom Implementation)

## 📦 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/VibeMatcher.git
   cd VibeMatcher
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run SpotifyVibeMatcher.py
   ```

## 🧠 The Algorithm
The core of VibeMatcher lies in its custom feature weighting. The algorithm prioritizes certain acoustic properties (like Acousticness and Speechiness) by applying a NumPy weight array before calculating the Euclidean distance. This ensures that the structural "feel" of a track is preserved even when hopping from Rap to Heavy Metal.