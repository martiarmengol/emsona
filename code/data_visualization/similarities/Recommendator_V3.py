from scipy.spatial.distance import cosine, euclidean
import pandas as pd
import numpy as np
import json
import pickle
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import unicodedata
import re
from sklearn.manifold import TSNE
from rapidfuzz import process, fuzz
import yt_dlp
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs
import os
import glob
import pickle
import argparse
import datetime
from pytube import YouTube
from pydub import AudioSegment
import shutil

def sanitize_title(title: str, channel: str) -> str:
    # Simple title cleaner – you can customize this further
    invalid_chars = r'<>:"/\|?*'
    for ch in invalid_chars:
        title = title.replace(ch, '')
    return f"{title.strip()}".replace(' ', '_')

def download_youtube_audio_mp3(youtube_url: str, output_folder_song: str, output_folder_metadata: str):
    """
    Downloads a single YouTube video's audio as MP3 into output_folder_song.
    Stores metadata in metadata.csv inside output_folder_metadata with format:
    video_id,song_title,channel_name,original_title,audio_file
    """
    os.makedirs(output_folder_song, exist_ok=True)
    os.makedirs(output_folder_metadata, exist_ok=True)
    
    clean_url = youtube_url.split('&')[0]

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_folder_song, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'geo_bypass': True,
        'force_ipv4': True,
        'nocheckcertificate': True,
        'hls_prefer_native': True,
        'http_chunk_size': 10485760,
        'http_headers': {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/112.0.0.0 Safari/537.36'
            ),
        },
        'quiet': False,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(clean_url, download=True)

        video_id = info.get("id", "")
        original_title = info.get("title", "")
        channel = info.get("uploader", "")
        clean_title = sanitize_title(original_title, channel)
        filename = f"{video_id}_{clean_title}.mp3"
        mp3_path = os.path.join(output_folder_song, filename)

        # Rename downloaded file
        downloaded_path = os.path.join(output_folder_song, f"{video_id}.mp3")
        if os.path.exists(downloaded_path):
            os.rename(downloaded_path, mp3_path)

        # Append metadata
        metadata_file = os.path.join(output_folder_metadata, "metadata.csv")
        new_entry = {
            'video_id': video_id,
            'song_title': clean_title,
            'channel_name': channel,
            'original_title': original_title,
            'audio_file': filename
        }

        if os.path.exists(metadata_file):
            df = pd.read_csv(metadata_file)
            if video_id not in df['video_id'].astype(str).tolist():
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                df.to_csv(metadata_file, index=False)
        else:
            pd.DataFrame([new_entry]).to_csv(metadata_file, index=False)

        print(f"Saved MP3 to: {mp3_path}")
        print(f"Metadata updated: {metadata_file}")
        return filename

def compute_effnet_embeddings_for_folder(
    folder: str,
    model_path: str,
    output_folder: str = None
) -> None:
    """
    Compute embeddings for every .mp3 in `folder` (including subdirectories)
    using the TensorFlow graph at `model_path`. Results are pickled to
    `<foldername>-<YYYY-MM-DD>-effnet.pkl` in `output_folder` (or script dir).
    """
    # Verify model exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Effnet model not found at: {model_path}")

    # Prepare output folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if output_folder is None:
        output_folder = base_dir
    os.makedirs(output_folder, exist_ok=True)

    # Build output filename
    date_str = datetime.date.today().isoformat()
    folder_basename = os.path.basename(os.path.normpath(folder))
    output_filename = "embedding_song.pkl"
    output_path = os.path.join(output_folder, output_filename)

    # Initialize the Essentia TF model
    model = TensorflowPredictEffnetDiscogs(
        graphFilename=model_path,
        output="PartitionedCall:1"
    )

    entries = []
    for file_path in sorted(glob.glob(os.path.join(folder, "**", "*.mp3"), recursive=True)):
        # Parse ID, song, artist from filename
        name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
        parts = name_no_ext.rsplit("-", 2)
        if len(parts) == 3:
            id_str, song_str, artist_str = parts
        elif len(parts) == 2:
            id_str, song_str = parts
            artist_str = "unknown"
        else:
            id_str = parts[0]
            song_str = ""
            artist_str = "unknown"

        # Load audio and compute embedding
        audio = MonoLoader(filename=file_path, sampleRate=16000, resampleQuality=4)()
        embedding = model(audio)
        embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

        entries.append({
            "id": id_str,
            "artist": artist_str,
            "song": song_str,
            "embedding": embedding_list,
        })

    # Save to pickle
    with open(output_path, "wb") as f:
        pickle.dump(entries, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Processed {len(entries)} files, saved embeddings to {output_path}")


def fuzzy_merge(df1, df2, key1, key2, threshold=90, limit=1):
    """
    df1[key1] will be matched to df2[key2] using fuzzy matching.
    Returns a DataFrame with matched rows from df2 and similarity score.
    """
    s = df2[key2].tolist()

    matches = df1[key1].apply(lambda x: process.extractOne(x, s, scorer=fuzz.token_sort_ratio))
    df1["best_match"] = [m[0] if m else None for m in matches]
    df1["score"] = [m[1] if m else None for m in matches]

    df_matched = df1[df1["score"] >= threshold]
    df_merged = pd.merge(df_matched, df2, left_on="best_match", right_on=key2, how="left")
    return df_merged

# --- Loaders for EffNet (JSON) and Maest (PKL) ---
def load_effnet_embeddings(path, population_label):
    with open(path, 'r') as f:
        data = json.load(f)
    rows = []
    for entry in data:
        artist = entry["artist"]
        song = entry["song"]
        embedding_matrix = np.array(entry["embedding"])
        agg_embedding = np.mean(embedding_matrix, axis=0)
        rows.append({
            "Song": song,
            "Artist": artist,
            "Population": population_label,
            "Embedding": agg_embedding
        })
    return rows




# --- Recommendation function ---
def recommend_similar_songs(df, query_id, n=5, k=12, metric='cosine', query_embedding=None):
    """
    Recommend N most similar songs to the query song using either cosine or Euclidean distance.
    If the query song is not found in the DataFrame, use the provided embedding to compute similarity.

    Parameters:
    - df: DataFrame with songs and their embeddings.
    - query_id: string in the format "Artist::Song".
    - n: number of recommendations to return.
    - k: number of clusters (used to limit comparisons within the same cluster).
    - metric: 'cosine' or 'euclidean'.
    - query_embedding: embedding vector for the query song (if not already in df).

    Returns:
    - DataFrame with top-N recommended songs from the same cluster.
    """
    assert metric in ['cosine', 'euclidean'], "Metric must be 'cosine' or 'euclidean'"
    assert f"Cluster_{k}" in df.columns, f"Cluster_{k} not found in DataFrame"

    try:
        artist, song = query_id.split("::")
    except ValueError:
        raise ValueError("query_id must be in the format 'Artist::Song'")

    query_row = df[(df["Artist"] == artist) & (df["Song"] == song)]
    embedding_cols = [col for col in df.columns if col.startswith("e")]

    if not query_row.empty:
        # Query song found in DataFrame
        query_embedding_vec = query_row[embedding_cols].values[0]
        query_cluster = query_row[f"Cluster_{k}"].values[0]
        cluster_df = df[df[f"Cluster_{k}"] == query_cluster].copy()
        cluster_df = cluster_df[~((cluster_df["Artist"] == artist) & (cluster_df["Song"] == song))]
    else:
        # Query song not found, use provided embedding
        if query_embedding is None:
            raise ValueError(f"Song '{query_id}' not found in DataFrame and no embedding provided.")

        query_embedding_vec = query_embedding
        kmeans = KMeans(n_clusters=k, random_state=42).fit(df[embedding_cols])
        query_cluster = kmeans.predict([query_embedding_vec])[0]
        cluster_df = df[df[f"Cluster_{k}"] == query_cluster].copy()

    def similarity(row):
        candidate_embedding = row[embedding_cols].values
        return 1 - cosine(query_embedding_vec, candidate_embedding) if metric == 'cosine' else euclidean(query_embedding_vec, candidate_embedding)

    cluster_df["Similarity"] = cluster_df.apply(similarity, axis=1)
    sorted_df = cluster_df.sort_values(by="Similarity", ascending=(metric == 'euclidean'))
    return sorted_df[["Artist", "Song", "Similarity", "YT Link"]].head(n)

# --- Plotting function ---
def plot_recommendations(df, query_id, recommendations, k):
    fig = px.scatter(
        df, x="x", y="y", color=df[f"Cluster_{k}"].astype(str), hover_name="Label",
        title=f"Query & Recommendations for {query_id} (k={k})"
    )

    query_artist, query_song = query_id.split("::")
    query_mask = (df["Artist"] == query_artist) & (df["Song"] == query_song)
    rec_mask = df.apply(lambda row: f"{row['Artist']}::{row['Song']}" in
                        [f"{a}::{s}" for a, s in zip(recommendations["Artist"], recommendations["Song"])], axis=1)

    fig.add_scatter(x=df[query_mask]["x"], y=df[query_mask]["y"],
                    mode='markers+text', name='Query Song', marker=dict(size=14, color='black'),
                    text=["Query"], textposition="top center")

    fig.add_scatter(x=df[rec_mask]["x"], y=df[rec_mask]["y"],
                    mode='markers+text', name='Recommendations', marker=dict(size=10, color='red'),
                    text=recommendations["Song"], textposition="bottom center")

    fig.show()


def load_maest_embeddings(path, population_label):
    with open(path, "rb") as f:
        data = pickle.load(f)

    rows = []
    for entry in data:
        embedding_data = entry["embedding"]
        artist = entry["artist"]
        song = entry["song"]

        flattened = []
        for i in range(min(len(embedding_data), 100)):
            if embedding_data[i]:
                flattened.extend(embedding_data[i])

        flattened = np.array(flattened[:1000])
        rows.append({
            "Song": song,
            "Artist": artist,
            "Population": population_label,
            "Embedding": flattened
        })
    return rows


# --- Choose model and load data ---
def build_dataframe(model="effnet"):
    if model == "effnet":
        print("1")
        before = load_maest_embeddings("/home/guillem/Music/MTG-102/code/essentia-models/maest/embeddings/before_2012-2025-05-08_maest.pkl", "Before 2012")
        print("2")
        after = load_maest_embeddings("/home/guillem/Music/MTG-102/code/essentia-models/maest/embeddings/after_2018-2025-05-08_maest.pkl", "After 2018")
        print("3")
        extra_song = load_maest_embeddings("/home/guillem/Pictures/Embedding/embedding_song.pkl", "Selected Song")
    else:
        raise ValueError("Model must be 'effnet'")
    
    all_songs = before + after + extra_song
    flat_data = []
    for song in all_songs:
        row = {
            "Song": song["Song"],
            "Artist": song["Artist"],
            "Population": song["Population"]
        }
        for i, val in enumerate(song["Embedding"]):
            row[f"e{i}"] = val
        flat_data.append(row)
    return pd.DataFrame(flat_data)

def query_id_creator(artist, song):
    return f"{artist}::{song}"

def normalize_text(text):
    text = text.replace("_", " ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))  # Remove accents
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text.strip().lower()


def extract_embedding(audio_path):
    # Pseudocode – adapt to your real embedding system
    audio = MonoLoader(filename=audio_path)()
    model = TensorflowPredictEffnetDiscogs(graphFilename="path/to/effnet.pb")
    embedding = model(audio)  # shape (T, D)
    agg_embedding = np.mean(embedding, axis=0)
    return agg_embedding


def clear_folders(folder1: str, folder2: str, folder3: str) -> None:
    """
    Removes all contents inside the two given folders.
    
    Args:
        folder1 (str): Path to the first folder.
        folder2 (str): Path to the second folder.
    """
    for folder in [folder1, folder2, folder3]:
        if not os.path.isdir(folder):
            print(f"⚠️ Folder does not exist: {folder}")
            continue
        
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"❌ Failed to delete {item_path}: {e}")
        print(f"✅ Cleared: {folder}")


if __name__ == "__main__":



    # INFORMATION TO BE EDITED
    #--------------------------------------------------------------------------------------------------------------------------------


    song_path = "/home/guillem/Pictures/Song"  #The folder where the song is gonna be downloaded
    song_url = "https://youtu.be/1wUTtl5gJMw"  #The url of the song we want to download
    embedding_path = "/home/guillem/Pictures/Embedding"  #The path of the embeddings we are using
    model_path = "/home/guillem/Music/MTG-102/code/essentia-models/effnet/discogs_multi_embeddings-effnet-bs64-1.pb"  #The path of the model we are using
    metadata_path = "/home/guillem/Music/MTG-102/youtube_playlist_scraper/catalan_music_metadata.csv"  #The path of the metadata of the songs we are using
    store_metadata_path = "/home/guillem/Pictures/Metadata"  # Where we want to store the metadata of the song we have just downloaded 
    
    metric = "euclidean"  #Netric we want to use (euclidean or cosine)
    k = 10  #Number of clusters we want to use
    n = 4  #Number of recommentations we want to get

    model = "effnet"  #DO NOT CHANGE!

    #--------------------------------------------------------------------------------------------------------------------------------


    store_metadata_path_file = store_metadata_path + "/metadata.csv"
    # 1. Download and embed the song
    filename = download_youtube_audio_mp3(song_url, song_path, store_metadata_path)


    compute_effnet_embeddings_for_folder(song_path, model_path, embedding_path )
    for i in range(10):
        print(filename)
    # 2. Load metadata
    metadata1 = pd.read_csv(metadata_path)

    metadata2 = pd.read_csv(store_metadata_path_file)

    metadata_df = pd.concat([metadata1, metadata2], ignore_index=True)
    metadata_df.columns = metadata_df.columns.str.strip()

    # 3. Load existing embeddings

    df = build_dataframe(model=model)

    # 4. Extract the downloaded song's name and use it as query
    downloaded_song_filename = os.path.splitext(os.path.basename(mp3_path))[0]
    parts = downloaded_song_filename.rsplit("-", 2)
    if len(parts) == 3:
        id_str, song_str, artist_str = parts
    elif len(parts) == 2:
        id_str, song_str = parts
        artist_str = "unknown"
    else:
        id_str = parts[0]
        song_str = ""
        artist_str = "unknown"

    # 5. Manually create a row and append it to df
    with open("/home/guillem/Pictures/Embedding/embedding_song.pkl", "rb") as f:
        song_data = pickle.load(f)
    song_embedding = song_data[0]["embedding"]
    flattened = []
    for i in range(min(len(song_embedding), 100)):
        if song_embedding[i]:
            flattened.extend(song_embedding[i])
    flattened = np.array(flattened[:1000])

    new_row = {
        "Song": song_str,
        "Artist": artist_str,
        "Population": "Selected Song"
    }
    for i, val in enumerate(flattened):
        new_row[f"e{i}"] = val
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # 6. Normalize for fuzzy merge
    df["query_key"] = (df["Artist"].str.replace("_", " ") + " " + df["Song"].str.replace("_", " ")).apply(normalize_text)
    metadata_df["meta_key"] = (metadata_df["Band"] + " " + metadata_df["Song Name"]).apply(normalize_text)

    # 7. Merge YT Links
    matched = fuzzy_merge(df, metadata_df[["meta_key", "YT Link"]], "query_key", "meta_key", threshold=85)
    df = matched

    # 8. Construct query ID
    query_id = query_id_creator(artist_str, song_str)

    # 9. Clustering
    embedding_cols = [col for col in df.columns if col.startswith("e")]
    X = df[embedding_cols].values
    kmeans = KMeans(n_clusters=k, random_state=42)
    df[f"Cluster_{k}"] = kmeans.fit_predict(X)

    # 10. Recommendations
    top_recs = recommend_similar_songs(df, query_id, n=n, k=k, metric=metric, query_embedding=flattened)    
    print(top_recs)

    # 11. t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X)
    df["x"] = X_2d[:, 0]
    df["y"] = X_2d[:, 1]
    df["Label"] = df["Artist"] + " - " + df["Song"]

    # 12. Plot
    plot_recommendations(df, query_id, top_recs, k=k)
    clear_folders(song_path, embedding_path, store_metadata_path)
