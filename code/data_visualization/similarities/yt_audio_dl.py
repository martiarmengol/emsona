#!/usr/bin/env python3
"""
YouTube Audio Downloader Script
Downloads high-quality MP3 audio from YouTube videos or playlists using yt-dlp.
"""
import os
import re
import sys
import argparse
import yt_dlp  # yt-dlp library for downloading YouTube content

# --- Configuration and Utility Function ---
def download_youtube_audio(url: str, output_dir: str = "."):
    """
    Download the best-quality audio from a YouTube video or playlist URL and convert to MP3.
    Returns a list of tuples: (video_id, song_title, channel_name, original_title, audio_file_name)
    """
    results = []  # To store metadata for each downloaded track
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # yt-dlp options for best audio and MP3 conversion
    ydl_opts = {
        'format': 'bestaudio/best',       # select the best audio quality format available
        'ignoreerrors': True,            # skip unavailable videos (continue playlist even if one fails)
        'quiet': True,                   # run quietly (suppress verbose output)
        'no_warnings': True,
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),  # temp filename as videoID.ext
        'postprocessors': [{             # use FFmpeg to convert the audio to MP3
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '0'     # '0' = best quality (highest bitrate available):contentReference[oaicite:7]{index=7}
        }],
        'overwrites': True,             # overwrite files if they exist (re-download support)
        'continuedl': False             # do not resume partially downloaded files; start fresh
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)  # download video or playlist info
    except Exception as e:
        print(f"Error: failed to download from URL {url}\n{e}", file=sys.stderr)
        return results

    # If a playlist is given, 'info' will be a dict with an 'entries' list
    entries = info.get('entries', None)
    if entries is None:
        entries = [info]  # single video info wrapped in a list for uniform processing

    for entry in entries:
        if entry is None:
            # Skip if video was unavailable (yt-dlp returns None in entries when ignoreerrors=True)
            continue
        video_id = entry.get("id")
        original_title = entry.get("title") or ""
        channel_name = entry.get("channel") or entry.get("uploader") or ""
        # Derive song_title by removing channel name (if it prefixes the title)
        song_title = original_title
        if channel_name and song_title.startswith(channel_name):
            # Remove the channel name and any leading dash or colon after it
            song_title = song_title[len(channel_name):]
            if song_title.startswith(" - ") or song_title.startswith(": "):
                song_title = song_title[3:]
            elif song_title.startswith("- ") or song_title.startswith(": "):
                song_title = song_title[2:]
        song_title = song_title.strip()  # trim any leftover whitespace
        
        # Replace spaces with underscores for the song_title (preserve other punctuation)
        song_title = song_title.replace(" ", "_")
        # Sanitize filesystem-unsafe characters (replace with underscore)
        song_title = re.sub(r'[\\/:*?"<>|]', '_', song_title)
        # Collapse multiple underscores (from replacements) into a single underscore
        song_title = re.sub(r'__+', '_', song_title)

        # Construct the final MP3 filename and path
        audio_file_name = f"{video_id}_{song_title}.mp3"
        audio_path = os.path.join(output_dir, audio_file_name)
        # The file was downloaded as <video_id>.mp3 (because of our outtmpl and postprocessor)
        temp_path = os.path.join(output_dir, f"{video_id}.mp3")
        try:
            if os.path.exists(temp_path):
                os.replace(temp_path, audio_path)  # rename to the final filename
            else:
                # If the temp file is not found, attempt to find and rename any file with the video_id
                for ext in ("webm", "m4a", "opus", "mp4"):
                    temp_alt = os.path.join(output_dir, f"{video_id}.{ext}")
                    if os.path.exists(temp_alt):
                        os.replace(temp_alt, audio_path)
                        break
        except OSError as err:
            print(f"Warning: could not rename file for video {video_id}: {err}", file=sys.stderr)

        # Append the result (video_id, song_title, channel_name, original_title, file_name)
        results.append((video_id, song_title, channel_name, original_title, audio_file_name))
    return results

# --- Command-line interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download high-quality MP3 audio from YouTube videos or playlists.")
    parser.add_argument("url", help="YouTube video URL or playlist URL to download audio from")
    parser.add_argument("-o", "--output-dir", default=".", help="Output directory for MP3 files (default: current directory)")
    args = parser.parse_args()


    for i in range(100):
        downloads = download_youtube_audio(args.url, args.output_dir)
        # Print out the metadata for each downloaded audio (tab-separated fields)
        for video_id, song_title, channel, orig_title, file_name in downloads:
            print(f"{video_id}\t{song_title}\t{channel}\t{orig_title}\t{file_name}")