"""
Golden-pair scorecard: does the algorithm agree with a human ear?

Each pair is (song A, song B) that YOU believe share a vibe across genres.
For each pair we print where B lands in A's results (1 = top match), with genre
ignored. Lower is better. Run it before and after changing weights:

    python evaluate_vibes.py

Replace the placeholder pairs below with your own picks (aim for 10+).
"""
import numpy as np

from recommender import DEFAULT_WEIGHTS, MusicRecommender, load_data

GOLDEN_PAIRS = [
    # (search for song A, search for song B)  - placeholders, swap in your own
    ("magnolia playboi carti", "the beautiful people marilyn manson"),
    ("x gon' give it to ya dmx", "break stuff limp bizkit"),
    ("sicko mode travis", "bodies drowning pool"),
    ("killing in the name rage", "party up dmx"),
]


def scorecard(recommender, pairs=GOLDEN_PAIRS):
    ranks = []
    for a, b in pairs:
        rank = recommender.rank_of(a, b)
        ranks.append(rank)
        shown = f"{rank:,}" if rank else "song not found"
        print(f"  {a[:30]:30} -> {b[:36]:36} rank {shown}")
    found = [r for r in ranks if r]
    if found:
        total = len(recommender.df)
        print(f"  median rank: {int(np.median(found)):,} of {total:,} songs "
              f"(top {np.median(found) / total:.1%})")
    return ranks


if __name__ == "__main__":
    print("Weights:", DEFAULT_WEIGHTS)
    scorecard(MusicRecommender(load_data()))
