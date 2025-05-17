#!/usr/bin/env python3
import os
import argparse
import time
import datetime
import pickle
from google import genai

"""
RUN THIS ON THE CONSOLE BEFORE EXECUTING THE SCRIPT(WITH YOUR ACTUAL API KEY):
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
        "4. Make sure the lyrics are coherent.\n"
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
        description="Extract lyrics from .mp3 files in a folder using Gemini API"
    )
    parser.add_argument("input_folder", help="Path to folder containing .mp3 files")
    parser.add_argument(
        "output_folder", help="Path to folder where lyrics.pkl will be saved"
    )
    args = parser.parse_args()

    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Please set the GENAI_API_KEY environment variable")
    client = genai.Client(api_key=api_key)

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
    sleep_time = 60.0 / 14.0  # Max 14 requests per minute
    max_retries = 10  # Maximum retry attempts on error

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

    # construct output filename with folder name and current date
    folder_name = os.path.basename(os.path.normpath(args.input_folder))
    date_str = datetime.date.today().isoformat()
    output_filename = f"{folder_name}-{date_str}-lyrics.pkl"
    output_path = os.path.join(args.output_folder, output_filename)
    with open(output_path, "wb") as out_file:
        pickle.dump(results, out_file)

    print(f"Saved lyrics for {len(results)} songs to '{output_path}'")


if __name__ == "__main__":
    main()
