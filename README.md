# 🎵 VibeMatcher Pro

A machine learning-based music recommendation engine built with Python and Streamlit. 

Unlike traditional recommendation systems that rely solely on genres or collaborative filtering, VibeMatcher analyzes the **acoustic DNA** of tracks (such as BPM, energy, danceability, and acousticness) to find mathematical similarities and suggest songs that share the exact same "vibe".

## ✨ Features
* **Acoustic Feature Analysis:** Uses K-Nearest Neighbors (KNN) logic to calculate vector distances between tracks based on their core audio features.
* **Smart Genre Mapping & Pre-Filtering:** Implements a custom waterfall algorithm and substring-matching pre-filters to clean, group, and normalize thousands of messy Spotify genres into accurate, user-friendly categories.
* **BPM Normalization:** Features tempo integration for highly accurate rhythm matching without skewing the mathematical distance.
* **Smart Filtering:** Automatically removes source-song duplicates (like remasters or alternative versions) from the recommendation list.
* **Interactive UI:** A clean, responsive Right-To-Left (RTL) interface built with Streamlit, complete with embedded Spotify web players.

## 🛠️ Tech Stack
* **Python** (Core Logic)
* **Pandas & NumPy** (Data Engineering, Data Wrangling & Vector Math)
* **Streamlit** (Frontend Web UI)

## ⚙️ How It Works
1. Enter a song you like.
2. Select a target genre (or let the system surprise you with the "Random" option).
3. The engine vectorizes the source song's acoustic traits and searches the target genre for the closest mathematical neighbors.
4. Discover 5 new tracks that match your exact vibe, playable directly from the browser.

## 📊 Data Source & Limitations
Please note that this application runs on a static, pre-processed dataset (provided via CSV files: `tracks.csv` and `artists.csv`) containing a specific subset of tracks and artists. 

It does **not** connect to the live Spotify API to fetch real-time releases or the entire Spotify library. If a specific song is not present in the underlying CSV dataset, the engine will gracefully notify the user that the track cannot be found.

## 🚀 How to Run Locally
1. Clone this repository to your local machine.
2. Ensure you have the datasets (`tracks.csv` and `artists.csv`) in the same directory.
3. Install the required dependencies:
   ```bash
   pip install streamlit pandas numpy