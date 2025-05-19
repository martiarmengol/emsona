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
        return clean_title, channel, mp3_path

def compute_effnet_embeddings_for_folder(
    folder: str,
    model: str,
    output_folder: str = None,
    song_name: str = None,
    channel_name: str = None
) -> None:
    """
    Compute embeddings for every .mp3 in folder (including subdirectories) and write to pickle.
    Output filename is based on the folder name, current date (YYYY-MM-DD), and effnet, with format <foldername>-<date>-effnet.pkl.
    Each entry in the pickle contains id, artist, song, and embedding.
    """
    import ast

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = model
    if output_folder is None:
        output_folder = base_dir
    os.makedirs(output_folder, exist_ok=True)

    # Output filename
    output_filename = "embedding_song.pkl"
    output_path = os.path.join(output_folder, output_filename)

    # Load model
    model = TensorflowPredictEffnetDiscogs(
        graphFilename=model_path, output="PartitionedCall:1"
    )

    entries = []
    for file_path in sorted(glob.glob(os.path.join(folder, "*.mp3"), recursive=True)):
        filename = os.path.basename(file_path)
        name_no_ext = os.path.splitext(filename)[0]

        # Extract id and assign consistent song/channel
        id_str = name_no_ext.split("_")[0]
        song_str = song_name.replace(' ', '_').replace('"', '').strip() if song_name else name_no_ext.replace(' ', '_').replace('"', '').strip()
        artist_str = channel_name.replace(' ', '_').replace('"', '').strip() if channel_name else "unknown"

        audio = MonoLoader(filename=file_path, sampleRate=16000, resampleQuality=4)()
        embedding = model(audio)
        embedding = np.mean(embedding, axis=0)

        try:
            embedding_list = embedding.tolist()
        except AttributeError:
            embedding_list = list(embedding)

        entries.append({
            "video_id": id_str,
            "song_title": song_str,
            "channel_name": artist_str,
            "youtube_link": "",
            "embedding": embedding_list,
        })

    with open(output_path, "wb") as f:
        pickle.dump(entries, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Processed {len(entries)} files, saved embeddings to {output_path}")

# --- Updated fuzzy_merge to match new metadata format ---
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
    df[embedding_cols] = df[embedding_cols].astype(float)  # Ensure correct dtype

    if not query_row.empty:
        query_embedding_vec = query_row[embedding_cols].values[0].astype(float)
        query_cluster = query_row[f"Cluster_{k}"].values[0]
        cluster_df = df[df[f"Cluster_{k}"] == query_cluster].copy()
        cluster_df = cluster_df[~((cluster_df["Artist"] == artist) & (cluster_df["Song"] == song))]
    else:
        if query_embedding is None:
            raise ValueError(f"Song '{query_id}' not found in DataFrame and no embedding provided.")
        query_embedding_vec = np.array(query_embedding, dtype=float)
        kmeans = KMeans(n_clusters=k, random_state=42).fit(df[embedding_cols])
        query_cluster = kmeans.predict([query_embedding_vec])[0]
        cluster_df = df[df[f"Cluster_{k}"] == query_cluster].copy()

    # Compute similarity/distance properly
    def similarity(row):
        candidate_embedding = np.array([row[col] for col in embedding_cols], dtype=float)
        return 1 - cosine(query_embedding_vec, candidate_embedding) if metric == 'cosine' else euclidean(query_embedding_vec, candidate_embedding)

    cluster_df["Similarity"] = cluster_df.apply(similarity, axis=1)

    # Sort direction depends on metric
    ascending = (metric == 'euclidean')
    sorted_df = cluster_df.sort_values(by="Similarity", ascending=ascending)

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


def load_embeddings(path, population_label):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    rows = []
    for entry in data:
        # Obtener ID del embedding
        entry_id = entry["video_id"]
        artist = entry["channel_name"]
        song = entry["song_title"]
        embedding_matrix = np.array(entry["embedding"])
        
        # Verificar si el embedding es ya un vector o una matriz
        if len(embedding_matrix.shape) > 1:
            # Si es una matriz, calcular la media a lo largo del eje 0
            agg_embedding = np.mean(embedding_matrix, axis=0)
        else:
            # Si ya es un vector, usarlo directamente
            agg_embedding = embedding_matrix
            
        rows.append({
            "ID": entry_id,  # Usar ID del embedding
            "Song": song,
            "Artist": artist,
            "Population": population_label,
            "Embedding": agg_embedding
        })
    return rows


# --- Choose model and load data ---
def build_dataframe(songs_embeddings, new_song_embeddings):
    songs = load_embeddings(song_embeddings, "All Songs")
    extra_song = load_embeddings(new_song_embeddings, "Selected Song")

    all_songs = extra_song + songs
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

# --- Updated query_id_creator to keep consistency with metadata ---
def query_id_creator(artist, song):
    artist = artist.replace(' ', '_').strip()
    song = song.replace(' ', '_').strip()
    return f"{artist}::{song}"


# --- Updated normalize_text to handle metadata format ---
def normalize_text(text):
    text = text.replace("_", " ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", "", text)
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


def recomendator(song_path, song_url, embedding_path, model_path, metadata_path, store_metadata_path, metric, song_embeddings, k = 1, n = 3):
    store_metadata_path_file = store_metadata_path + "/metadata.csv"
    new_song_embeddings = embedding_path + "/embedding_song.pkl"

    # Step 1: Download and embed the song
    song_name, channel_name, mp3_path = download_youtube_audio_mp3(song_url, song_path, store_metadata_path)
    compute_effnet_embeddings_for_folder(song_path, model_path, embedding_path, song_name, channel_name)

    # Step 2: Load metadata
    metadata1 = pd.read_csv(metadata_path)
    metadata2 = pd.read_csv(store_metadata_path_file)
    metadata_df = pd.concat([metadata1, metadata2], ignore_index=True)
    metadata_df.columns = metadata_df.columns.str.strip()

    # Step 3: Load embeddings
    df = build_dataframe(song_embeddings , new_song_embeddings)

    # Step 4: Normalize for fuzzy matching
    df["query_key"] = (df["Artist"] + " " + df["Song"]).apply(normalize_text)
    metadata_df["meta_key"] = (metadata_df["channel_name"] + " " + metadata_df["song_title"]).apply(normalize_text)

    # Step 5: Prepare YT links from video_id
    meta_cols = metadata_df[["meta_key", "video_id"]].copy()
    meta_cols["YT Link"] = "https://www.youtube.com/watch?v=" + meta_cols["video_id"]

    # Step 6: Fuzzy merge metadata
    matched = fuzzy_merge(df, meta_cols, "query_key", "meta_key", threshold=85)
    df = matched

    # Step 7: Generate query ID and cluster
    song_name = song_name.replace('_', ' ').strip()
    channel_name = channel_name.strip()
    query_id = query_id_creator(channel_name, song_name)

    embedding_cols = [col for col in df.columns if col.startswith("e")]
    df[embedding_cols] = df[embedding_cols].astype(float)  # ✅ Ensure float precision

    # Cluster
    X = df[embedding_cols].values
    kmeans = KMeans(n_clusters=k, random_state=42)
    df[f"Cluster_{k}"] = kmeans.fit_predict(X)

    # Step 8: Recommend similar songs
    top_recs = recommend_similar_songs(df, query_id, n=n, k=k, metric=metric)

    # Optional: Save full CSV
    # Step 9: t-SNE and plot
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X)

    # ✅ Add new columns efficiently to avoid fragmentation
    df = pd.concat([
        df,
        pd.DataFrame({
            "x": X_2d[:, 0],
            "y": X_2d[:, 1],
            "Label": df["Artist"] + " - " + df["Song"]
        }, index=df.index)
    ], axis=1)

    # Plot
    plot_recommendations(df, query_id, top_recs, k=k)

    # Step 10: Clear intermediate files
    clear_folders(song_path, embedding_path, store_metadata_path)
    return top_recs


if __name__ == "__main__":
    # --- PARAMETERS TO EDIT ---
    song_path = "/home/guillem/Pictures/Song"
    song_url = "https://youtu.be/fBUEkAR7ZCQ"
    embedding_path = "/home/guillem/Pictures/Embedding"
    model_path = "/home/guillem/Downloads/discogs_artist_embeddings-effnet-bs64-1.pb"
    metadata_path = "/home/guillem/Music/emsona/youtube_playlist_scraper/catalan_music_metadata.csv"
    store_metadata_path = "/home/guillem/Pictures/Metadata"
    store_metadata_path_file = store_metadata_path + "/metadata.csv"
    song_embeddings = "/home/guillem/Downloads/canciones1-2025-05-18-effnet-artist.pkl"

    metric = "euclidean"  # "cosine" or "euclidean"
    k = 1
    n = 3
    # --- END PARAMETERS ---
    # --- Run the recomendator ---
    top_recs = recomendator(song_path, song_url, embedding_path, model_path, metadata_path, store_metadata_path, metric, song_embeddings, k, n)
    print("Top recommendations:", top_recs)
    # --- END RUN ---
    #top_recs.to_csv("/home/guillem/Downloads/top_recommendations.csv", index=False)
    