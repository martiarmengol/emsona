import pandas as pd
import yt_dlp

# Load the CSV
df = pd.read_csv('after-18.csv') 

# yt-dlp options
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': '%(title)s.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}

for idx, row in df.iterrows():
    try:
        url = row['YT Link'].split('&')[0]  # Clean the URL
        print(f"Downloading {url}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"Successfully downloaded row {idx}")
    except Exception as e:
        print(f"Failed to process row {idx} ({url}): {e}")
