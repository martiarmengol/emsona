#!/usr/bin/env python3
import os
import csv
import argparse
import sys

def sanitize(text: str) -> str:
    """
    Convert text to lowercase, remove " - topic" and " oficial", strip whitespace.
    """
    text = text.lower()
    # remove unwanted substrings
    text = text.replace(' - topic', '')
    text = text.replace(' oficial', '')
    return text.strip()

def load_metadata(csv_path: str) -> dict:
    """
    Read the CSV and return a mapping from video_id to (channel_name, song_title).
    """
    mapping = {}
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row.get('video_id', '').strip()
                channel = row.get('channel_name', '').strip()
                title = row.get('song_title', '').strip()
                if not vid or not channel or not title:
                    # skip incomplete rows
                    continue
                mapping[vid] = (channel, title)
    except Exception as e:
        print(f"Error reading metadata file: {e}", file=sys.stderr)
        sys.exit(1)
    return mapping

def main():
    parser = argparse.ArgumentParser(
        description="Rename song files based on YouTube metadata"
    )
    parser.add_argument(
        "directory",
        help="Path to the folder containing song files to rename"
    )
    args = parser.parse_args()
    target_dir = args.directory

    if not os.path.isdir(target_dir):
        print(f"Error: '{target_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # locate CSV next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "catalan_music_metadata.csv")
    if not os.path.isfile(csv_path):
        print(f"Error: metadata file not found at '{csv_path}'.", file=sys.stderr)
        sys.exit(1)

    metadata = load_metadata(csv_path)

    # iterate files in target directory
    for filename in os.listdir(target_dir):
        orig_path = os.path.join(target_dir, filename)
        if not os.path.isfile(orig_path):
            continue

        renamed = False
        for vid, (channel, title) in metadata.items():
            if vid in filename:
                ext = os.path.splitext(filename)[1]
                new_base = f"{sanitize(channel)}-{sanitize(title)}"
                new_name = new_base + ext
                new_path = os.path.join(target_dir, new_name)

                if filename == new_name:
                    print(f"Skipping '{filename}': already correctly named.")
                elif os.path.exists(new_path):
                    print(f"Warning: target name '{new_name}' already exists; skipping '{filename}'.")
                else:
                    os.rename(orig_path, new_path)
                    print(f"Renamed '{filename}' → '{new_name}'")
                renamed = True
                break

        if not renamed:
            # no matching video_id
            continue

if __name__ == "__main__":
    main()
