#!/usr/bin/env python3
"""
Filter non-Catalan lyrics and embeddings, generate CSV of deleted songs.

Usage:
  pip install langdetect
  python3 filter_catalan.py

This script:
- Loads the lyrics PKL (list of dicts with 'id' and 'lyrics').
- Detects language probabilities of each lyrics entry using langdetect.
- Removes entries whose Catalan probability is <= 0.5.
- Loads effnet embeddings PKLs (multi and artist), each a list of dicts with 'video_id'.
- Filters out entries whose video_id is among removed IDs.
- Writes a CSV of removed songs with columns: id, song_title, channel_name, youtube_link, lyrics.
- Overwrites the three PKL files with the filtered data.
"""
import os
import pickle
import csv
from langdetect import detect_langs, DetectorFactory

# Ensure reproducible detection
DetectorFactory.seed = 0

# Paths (relative to project root)
LYRICS_PKL = "code/lyrics_extractor/canciones1-2025-05-19-lyrics.pkl"
MULTI_PKL = (
    "code/essentia-models/effnet/embeddings/canciones1-2025-05-17-effnet-multi.pkl"
)
ARTIST_PKL = (
    "code/essentia-models/effnet/embeddings/canciones1-2025-05-18-effnet-artist.pkl"
)
DELETED_CSV = "code/lyrics_extractor/deleted_songs.csv"


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def main():
    # Load data
    lyrics_list = load_pkl(LYRICS_PKL)
    multi_list = load_pkl(MULTI_PKL)
    artist_list = load_pkl(ARTIST_PKL)

    # Detect entries where Catalan probability > 0.5
    removed_ids = []
    kept_lyrics = []
    for entry in lyrics_list:
        text = entry.get("lyrics", "").strip()
        try:
            langs = detect_langs(text) if text else []
            ca_prob = next((l.prob for l in langs if l.lang == "ca"), 0.0)
        except Exception:
            ca_prob = 0.0
        if ca_prob > 0.5:
            kept_lyrics.append(entry)
        else:
            removed_ids.append(entry.get("id"))

    # Map multi embeddings by id
    multi_map = {e.get("video_id"): e for e in multi_list}

    # Build removed rows for CSV
    removed_rows = []
    for rid in removed_ids:
        meta = multi_map.get(rid)
        if not meta:
            continue
        lyrics_text = next(
            (e.get("lyrics", "") for e in lyrics_list if e.get("id") == rid), ""
        )
        removed_rows.append(
            {
                "id": rid,
                "song_title": meta.get("song_title", ""),
                "channel_name": meta.get("channel_name", ""),
                "youtube_link": meta.get("youtube_link", ""),
                "lyrics": lyrics_text,
            }
        )

    # Write CSV of removed songs
    os.makedirs(os.path.dirname(DELETED_CSV), exist_ok=True)
    with open(DELETED_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["id", "song_title", "channel_name", "youtube_link", "lyrics"],
        )
        writer.writeheader()
        writer.writerows(removed_rows)

    # Filter embeddings lists
    kept_multi = [e for e in multi_list if e.get("video_id") not in removed_ids]
    kept_artist = [e for e in artist_list if e.get("video_id") not in removed_ids]

    # Overwrite PKLs
    save_pkl(kept_lyrics, LYRICS_PKL)
    save_pkl(kept_multi, MULTI_PKL)
    save_pkl(kept_artist, ARTIST_PKL)

    print(f"Removed {len(removed_ids)} entries.")
    print(f"CSV of deleted songs written to: {DELETED_CSV}")
    print("Updated PKL files with Catalan-only entries.")


if __name__ == "__main__":
    main()
