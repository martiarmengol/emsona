import os
import pandas as pd
import yt_dlp
import json
from urllib.parse import urlparse, parse_qs
import re

# CONFIGURATION
AUDIO_OUTPUT_DIR = 'downloads'
METADATA_OUTPUT_FILE = 'catalan_music_metadata.csv'
INPUT_CSV = 'playlists.csv'




# Ensure output folder exists
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

# Load list of playlists
df_playlists = pd.read_csv(INPUT_CSV)
playlist_urls = df_playlists['playlist_url'].dropna().tolist()

# Track downloaded video IDs
downloaded_ids = set()
if os.path.exists(METADATA_OUTPUT_FILE):
    existing_df = pd.read_csv(METADATA_OUTPUT_FILE)
    downloaded_ids.update(existing_df['video_id'].tolist())
else:
    existing_df = pd.DataFrame()

# Collect metadata
metadata_records = []

# YT-DLP options for metadata fetching
ydl_opts_info = {
    'quiet': True,
    'ignoreerrors': True,
    'extract_flat': True,
    'skip_download': True
}

# YT-DLP options for audio download
def get_ydl_audio_opts(output_filename):
    return {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(AUDIO_OUTPUT_DIR, f'{output_filename}.%(ext)s'),
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'ignoreerrors': True,
    }

# Helper: clean and standardize title

def clean_title(title, channel):
    if not title:
        title = "unknown_title"
    if not channel:
        channel = ""

    # Remove channel name from title
    title = title.replace(channel, '').strip()

    # Remove illegal/special characters
    import re
    title = re.sub(r'[\\/:"*?<>|]+', '', title)   # Windows-illegal characters
    title = re.sub(r'[’"“”‘]', '', title)         # Fancy quotes
    title = title.replace(' - ', ' ').replace('—', ' ')
    title = title.strip()

    # Replace spaces with underscores
    return '_'.join(title.split())[:100]  # Limit length

# Iterate over playlists
for playlist_url in playlist_urls:
    try:
        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
            if not info or 'entries' not in info:
                print(f"Failed to fetch playlist: {playlist_url}")
                continue

            for entry in info['entries']:
                if entry is None:
                    continue
                video_id = entry.get('id')
                title = entry.get('title')
                channel = entry.get('uploader')

                if not video_id or video_id in downloaded_ids:
                    continue

                # Clean metadata
                clean_song_title = clean_title(title, channel)
                filename = f"{video_id}_{clean_song_title}"

                # Download audio
                try:
                    with yt_dlp.YoutubeDL(get_ydl_audio_opts(filename)) as ydl:
                        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                except Exception as e:
                    print(f"Error downloading {video_id}: {e}")
                    continue

                # Add metadata record
                metadata_records.append({
                    'video_id': video_id,
                    'song_title': clean_song_title,
                    'channel_name': channel,
                    'original_title': title,
                    'audio_file': f"{filename}.mp3"
                })

                downloaded_ids.add(video_id)
    except Exception as e:
        print(f"Error processing playlist {playlist_url}: {e}")

# Save metadata
final_df = pd.concat([existing_df, pd.DataFrame(metadata_records)], ignore_index=True)
final_df.to_csv(METADATA_OUTPUT_FILE, index=False)
print(f"Saved metadata to {METADATA_OUTPUT_FILE}")
