#!/usr/bin/env python3
import os
import argparse
import time
import datetime
import pickle
from google import genai

"""
RUN THIS ON THE CONSOLE BEFORE EXECUTING THE SCRIPT (WITH YOUR ACTUAL API KEY):
export GENAI_API_KEY="your_actual_api_key"
"""


def get_lyrics(client, file_path):
    """
    Upload the audio file and request lyrics transcription from Gemini.
    """
    uploaded = client.files.upload(file=file_path)
    prompt = (
        "You are a lyrics transcription assistant. Your task is to transcribe the lyrics of the song \n"
        "from the audio file provided. Follow these steps:\n"
        f"0. This is the audio file path: {file_path}\n"
        "1. Listen to the audio file.\n"
        "2. The language is catalan.\n"
        "3. Transcribe the lyrics, and if the song is in spanish ignore the rest of the instructions and return the lyrics.\n"
        "4. make sure the lyrics are coherent.\n"
        "5. If possible, contrast the result with Google.\n"
        "6. Make sure the catalan lyrics are correct.\n"
        "7. Return the final lyrics."
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=[prompt, uploaded]
    )
    return response.text.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Extract or retry lyrics transcription using Gemini API"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Extract mode: scan folder of .mp3 and create a new .pkl with lyrics
    extract = subparsers.add_parser("extract", help="Extract lyrics from .mp3 files")
    extract.add_argument("input_folder", help="Path to folder containing .mp3 files")
    extract.add_argument(
        "output_folder", help="Path to folder where lyrics.pkl will be saved"
    )

    # Retry mode: load existing .pkl and retry missing lyrics
    retry = subparsers.add_parser(
        "retry", help="Retry missing lyrics in existing .pkl file"
    )
    retry.add_argument("retry_pkl", help="Path to existing lyrics.pkl file to retry")
    retry.add_argument(
        "audio_folder", help="Path to folder containing .mp3 files for retry"
    )

    args = parser.parse_args()

    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Please set the GENAI_API_KEY environment variable")
    client = genai.Client(api_key=api_key)

    # Rate limit controls
    sleep_time = 60.0 / 14.0  # Max 14 requests per minute
    max_retries = 10  # Maximum retry attempts on error

    if args.mode == "extract":
        # Validate folders
        if not os.path.isdir(args.input_folder):
            raise NotADirectoryError(
                f"Input folder '{args.input_folder}' not found or not a directory"
            )
        os.makedirs(args.output_folder, exist_ok=True)

        files = [
            fname
            for fname in os.listdir(args.input_folder)
            if fname.lower().endswith(".mp3")
        ]
        total = len(files)
        if total == 0:
            print(f"No .mp3 files found in '{args.input_folder}'")
            return

        results = []
        for idx, fname in enumerate(files, start=1):
            file_path = os.path.join(args.input_folder, fname)
            id_ = os.path.splitext(fname)[0]

            print(f"[{idx}/{total}] Processing id='{id_}'")
            lyrics = None
            for attempt in range(1, max_retries + 1):
                try:
                    lyrics = get_lyrics(client, file_path)
                    break
                except Exception as e:
                    print(f"Attempt {attempt}/{max_retries} for '{fname}' failed: {e}")
                    if attempt < max_retries:
                        time.sleep(sleep_time)
            if lyrics is None:
                print(
                    f"Failed to retrieve lyrics for '{fname}' after {max_retries} attempts"
                )
                continue

            results.append({"id": id_, "lyrics": lyrics})
            print(f"  Retrieved lyrics ({len(lyrics)} characters)")

            if idx < total:
                time.sleep(sleep_time)

        # Save new .pkl
        folder_name = os.path.basename(os.path.normpath(args.input_folder))
        date_str = datetime.date.today().isoformat()
        output_filename = f"{folder_name}-{date_str}-lyrics.pkl"
        output_path = os.path.join(args.output_folder, output_filename)
        with open(output_path, "wb") as out_file:
            pickle.dump(results, out_file)

        print(f"Saved lyrics for {len(results)} songs to '{output_path}'")

    elif args.mode == "retry":
        # Validate inputs
        if not os.path.isfile(args.retry_pkl):
            raise FileNotFoundError(f"PKL file '{args.retry_pkl}' not found")
        if not os.path.isdir(args.audio_folder):
            raise NotADirectoryError(
                f"Audio folder '{args.audio_folder}' not found or not a directory"
            )

        # Load existing results
        with open(args.retry_pkl, "rb") as in_file:
            results = pickle.load(in_file)

        total = len(results)
        updated = 0
        for idx, entry in enumerate(results, start=1):
            if entry.get("lyrics"):
                continue
            id_ = entry.get("id")
            fname = f"{id_}.mp3"
            file_path = os.path.join(args.audio_folder, fname)
            if not os.path.isfile(file_path):
                print(
                    f"[{idx}/{total}] Audio file for id='{id_}' not found at '{file_path}', skipping"
                )
                continue

            print(f"[{idx}/{total}] Retrying id='{id_}' missing lyrics")
            lyrics = None
            for attempt in range(1, max_retries + 1):
                try:
                    lyrics = get_lyrics(client, file_path)
                    break
                except Exception as e:
                    print(f"Attempt {attempt}/{max_retries} for '{fname}' failed: {e}")
                    if attempt < max_retries:
                        time.sleep(sleep_time)
            if lyrics:
                entry["lyrics"] = lyrics
                updated += 1
                print(f"  Retrieved lyrics ({len(lyrics)} characters)")
            else:
                print(
                    f"Failed to retrieve lyrics for '{fname}' after {max_retries} attempts"
                )

            if idx < total:
                time.sleep(sleep_time)

        # After retrying missing lyrics, add any new audio files not in results
        existing_ids = {entry["id"] for entry in results}
        audio_files = [
            fname
            for fname in os.listdir(args.audio_folder)
            if fname.lower().endswith(".mp3")
        ]
        for fname in audio_files:
            id_ = os.path.splitext(fname)[0]
            if id_ in existing_ids:
                continue
            file_path = os.path.join(args.audio_folder, fname)
            print(f"[new] Adding id='{id_}' from audio folder")
            lyrics = None
            for attempt in range(1, max_retries + 1):
                try:
                    lyrics = get_lyrics(client, file_path)
                    break
                except Exception as e:
                    print(f"Attempt {attempt}/{max_retries} for '{fname}' failed: {e}")
                    if attempt < max_retries:
                        time.sleep(sleep_time)
            if lyrics:
                results.append({"id": id_, "lyrics": lyrics})
                updated += 1
                print(f"  Retrieved lyrics ({len(lyrics)} characters)")
            else:
                print(
                    f"Failed to retrieve lyrics for '{fname}' after {max_retries} attempts"
                )
        # Overwrite existing .pkl with updates
        with open(args.retry_pkl, "wb") as out_file:
            pickle.dump(results, out_file)

        print(f"Updated lyrics for {updated}/{total} entries in '{args.retry_pkl}'")


if __name__ == "__main__":
    main()
