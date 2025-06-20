from scipy.spatial.distance import cosine, euclidean
import pandas as pd
import numpy as np
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
from pytube import YouTube
from pydub import AudioSegment
import shutil
import faiss
from sklearn.decomposition import PCA
import requests
import argparse


def sanitize_title(title: str, channel: str) -> str:
    # Simple title cleaner – you can customize this further
    invalid_chars = r'<>:"/\|?*'
    for ch in invalid_chars:
        title = title.replace(ch, '')
    return f"{title.strip()}".replace(' ', '_')

def download_youtube_audio_mp3(youtube_url: str, output_folder_song: str, output_folder_metadata: str) -> tuple:
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
        return [(clean_title, channel, mp3_path)]

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
def plot_recommendations(df, query_id, recommendations, k, output_path="recommendation_plot.svg"):
    """
    Reduce embeddings to 2D for visualization and save the plot as an SVG file.

    Args:
        df (pd.DataFrame): DataFrame containing song embeddings and metadata.
        query_id (str): Query song ID in the format "Artist::Song".
        recommendations (pd.DataFrame): DataFrame containing recommended songs.
        k (int): Number of clusters used for visualization.
        output_path (str): Path to save the SVG plot.
    """
    # Reduce embeddings to 2D for visualization
    embedding_cols = [col for col in df.columns if col.startswith("e")]
    pca = PCA(n_components=2)
    df[['x', 'y']] = pca.fit_transform(df[embedding_cols])

    # Create scatter plot
    fig = px.scatter(
        df, x="x", y="y", color=df[f"Cluster_{k}"].astype(str), hover_name=None
    )

    # Highlight query song and recommendations
    query_artist, query_song = query_id.split("::")
    query_mask = (df["Artist"] == query_artist) & (df["Song"] == query_song)
    rec_mask = df.apply(lambda row: f"{row['Artist']}::{row['Song']}" in
                        [f"{a}::{s}" for a, s in zip(recommendations["Artist"], recommendations["Song"])], axis=1)

    # Add query song (black dot without name)
    fig.add_scatter(x=df[query_mask]["x"], y=df[query_mask]["y"],
                    mode='markers', name=None, marker=dict(size=14, color='black'))

    # Add recommended songs (red dots without names)
    fig.add_scatter(x=df[rec_mask]["x"], y=df[rec_mask]["y"],
                    mode='markers', name=None, marker=dict(size=12, color='red'))

    # Remove title, legend, and axis labels
    fig.update_layout(
        title=None,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False
    )

    # Save the plot as an SVG file
    fig.write_image(output_path)
    print(f"✅ SVG plot saved to: {output_path}")


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
def build_dataframe(song_embeddings, new_song_embeddings):
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
    """
    Creates a standardized query ID from artist and song names.
    """
    # Remove special characters and normalize
    artist = normalize_text(artist)
    song = normalize_text(song)
    # Replace spaces with underscores for the final format
    artist = artist.replace(" ", "_")
    song = song.replace(" ", "_")
    return f"{artist}::{song}"


# --- Updated normalize_text to handle metadata format ---
def normalize_text(text):
    """
    Normalizes text by removing special characters, accents, and converting to lowercase.
    """
    if not isinstance(text, str):
        return ""
    # Replace underscores and hyphens with spaces
    text = text.replace("_", " ").replace("-", " ")
    # Normalize Unicode characters (remove accents)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Remove special characters but keep spaces
    text = re.sub(r"[^\w\s]", "", text)
    # Remove extra spaces and convert to lowercase
    return " ".join(text.split()).lower()


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


def recomendator(song_path, song_url, embedding_path, model_path, metadata_path, store_metadata_path, metric, song_embeddings, path_to_cookies, k = 1, n = 3):
    clear_folders(song_path, embedding_path, store_metadata_path)

    store_metadata_path_file = store_metadata_path + "/metadata.csv"
    new_song_embeddings = embedding_path + "/embedding_song.pkl"

    # Step 1: Download and embed the song
    song_name, channel_name, mp3_path = download_youtube_audio_mp3(song_url, song_path, store_metadata_path, path_to_cookies)
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
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')    
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


def find_closest_embeddings_faiss(embeddings, query_embedding, top_k=5):
    """
    Use FAISS to find the closest embeddings to the query embedding.
    """
    embedding_dim = embeddings.shape[1]
    embeddings = np.ascontiguousarray(embeddings, dtype='float32')  # 👈 FIX here
    query_embedding = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype='float32')

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    distances, indices = index.search(query_embedding, top_k)
    return indices[0], distances[0]

def download_mp3_with_api(youtube_url, output_folder_song, output_folder_metadata):
    """
    Downloads an MP3 file from a YouTube video using the youtube-mp36 API.
    Creates a metadata file in the specified metadata folder.

    Args:
        youtube_url (str): The URL of the YouTube video.
        output_folder_song (str): The folder where the MP3 file will be saved.
        output_folder_metadata (str): The folder where the metadata file will be saved.

    Returns:
        tuple: (song_title, channel_name, mp3_path)
    """
    # Extract video ID from the YouTube URL
    video_id = youtube_url.split("v=")[-1] if "v=" in youtube_url else youtube_url.split("/")[-1]

    # API connection
    url = "https://youtube-mp36.p.rapidapi.com/dl"
    querystring = {"id": video_id}
    headers = {
        "x-rapidapi-key": "882822f1d3mshc30fbdb2e90cf17p19bf98jsnf5ce38ba6cf1",
        "x-rapidapi-host": "youtube-mp36.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch MP3 download link. Status code: {response.status_code}")

    data = response.json()
    if data.get("status") != "ok":
        raise Exception(f"Failed to fetch MP3 download link: {data.get('msg', 'Unknown error')}")

    # Extract metadata and download link
    download_link = data.get("link")
    song_title = data.get("title", "unknown_title").replace(" ", "_").replace("/", "_")
    channel_name = data.get("author", "unknown_channel")  # Placeholder for channel name
    mp3_filename = f"{song_title}.mp3"

    # Ensure the output folders exist
    os.makedirs(output_folder_song, exist_ok=True)
    os.makedirs(output_folder_metadata, exist_ok=True)

    # Download the MP3 file
    mp3_path = os.path.join(output_folder_song, mp3_filename)
    try:
        mp3_response = requests.get(download_link)
        if mp3_response.status_code == 200:
            with open(mp3_path, "wb") as f:
                f.write(mp3_response.content)
            print(f"MP3 downloaded and saved as '{mp3_path}'")
        else:
            raise Exception(f"Failed to download MP3. Status code: {mp3_response.status_code}")
    except Exception as e:
        raise Exception(f"Unexpected error while downloading MP3: {e}")

    # Create or update metadata file
    metadata_file = os.path.join(output_folder_metadata, "metadata.csv")
    new_entry = {
        'video_id': video_id,
        'song_title': song_title,
        'channel_name': channel_name,
        'original_title': data.get("title", "unknown_title"),
        'audio_file': mp3_filename
    }

    if os.path.exists(metadata_file):
        df = pd.read_csv(metadata_file)
        if video_id not in df['video_id'].astype(str).tolist():
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            df.to_csv(metadata_file, index=False)
    else:
        pd.DataFrame([new_entry]).to_csv(metadata_file, index=False)

    print(f"Metadata updated: {metadata_file}")
    return [(song_title, channel_name, mp3_path)]




def download_mp3_with_ytdlp(youtube_url, output_folder_song, output_folder_metadata):
    """
    Downloads an MP3 file from a YouTube video or playlist using yt-dlp.
    Creates a metadata file in the specified metadata folder.

    Args:
        youtube_url (str): The URL of the YouTube video or playlist.
        output_folder_song (str): The folder where the MP3 file(s) will be saved.
        output_folder_metadata (str): The folder where the metadata file will be saved.

    Returns:
        list of tuples: Each tuple is (song_title, channel_name, mp3_path)
    """
    os.makedirs(output_folder_song, exist_ok=True)
    os.makedirs(output_folder_metadata, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'outtmpl': os.path.join(output_folder_song, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '0'
        }],
        'overwrites': True
    }


    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)

    entries = info.get("entries", None) if info and 'entries' in info else [info]

    metadata_rows = []
    metadata_file = os.path.join(output_folder_metadata, "metadata.csv")

    for entry in entries:
        if entry is None:
            continue

        video_id = entry.get("id")
        original_title = entry.get("title") or "unknown_title"
        channel_name = entry.get("channel") or entry.get("uploader") or "unknown_channel"

        # Derive song_title from original_title (minus channel name if present)
        song_title = original_title
        if channel_name and song_title.startswith(channel_name):
            song_title = song_title[len(channel_name):].lstrip("-: ").strip()

        # Clean file name
        song_title_clean = re.sub(r'[\\/*?:"<>|]', '', song_title.replace(" ", ""))
        mp3_filename = f"{video_id}_{song_title_clean}.mp3"
        temp_path = os.path.join(output_folder_song, f"{video_id}.mp3")
        mp3_path = os.path.join(output_folder_song, mp3_filename)

        # Rename the downloaded file
        if os.path.exists(temp_path):
            os.replace(temp_path, mp3_path)

        # Update metadata
        new_entry = {
            'video_id': video_id,
            'song_title': song_title,
            'channel_name': channel_name,
            'original_title': original_title,
            'audio_file': mp3_filename
        }
        metadata_rows.append(new_entry)

        print(f"Downloaded: {mp3_filename}")

    # Append to or create metadata.csv
    if os.path.exists(metadata_file):
        df = pd.read_csv(metadata_file)
        existing_ids = df['video_id'].astype(str).tolist()
        new_rows = [row for row in metadata_rows if row['video_id'] not in existing_ids]
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            df.to_csv(metadata_file, index=False)
    else:
        pd.DataFrame(metadata_rows).to_csv(metadata_file, index=False)

    return [(song_title, channel_name, mp3_path)]

import math
import plotly.graph_objects as go

def plot_svg_recommendations(query_id, recommendations, output_path="recommendation_plot.svg"):
    """
    Plots the query song in the center and the N recommendations around it in a circle.
    Saves the result as an SVG image.
    """
    # Parse artist and song
    query_artist, query_song = query_id.split("::")

    # Center coordinates for the query
    x_query, y_query = 0, 0
    num_recs = len(recommendations)

    # Calculate circular layout for recommendations
    angle_step = 2 * math.pi / num_recs if num_recs > 0 else 0
    radius = 1.5  # distance from the center

    x_coords = [x_query]
    y_coords = [y_query]
    labels = [f"{query_artist} - {query_song} (Query)"]
    colors = ['black']

    for i, row in recommendations.iterrows():
        angle = i * angle_step
        x = x_query + radius * math.cos(angle)
        y = y_query + radius * math.sin(angle)
        label = f"{row['Artist']} - {row['Song']}\nDist: {row['Distance']:.2f}" if 'Distance' in row else f"{row['Artist']} - {row['Song']}"
        x_coords.append(x)
        y_coords.append(y)
        labels.append(label)
        colors.append('red')

    fig = go.Figure()

    # Add points
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers+text',
        marker=dict(size=14, color=colors),
        text=labels,
        textposition="top center"
    ))

    fig.update_layout(
        title=f"Recommendations for: {query_artist} - {query_song}",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False
    )

    fig.write_image(output_path)
    print(f"✅ SVG plot saved to: {output_path}")


def recomendator_with_faiss(song_path, song_url, embedding_path, model_path, metadata_path, store_metadata_path, song_embeddings, store_svg, k=1, n=5):
    store_metadata_path_file = os.path.join(store_metadata_path, "metadata.csv")
    new_song_embeddings = os.path.join(embedding_path, "embedding_song.pkl")

    # Step 1: Download and embed the song
    try:
        clear_folders(song_path, embedding_path, store_metadata_path)
        song_name, channel_name, mp3_path = download_mp3_with_ytdlp(song_url, song_path, store_metadata_path)[0]
    except Exception:
        song_name, channel_name, mp3_path = download_youtube_audio_mp3(song_url, song_path, store_metadata_path)

    compute_effnet_embeddings_for_folder(song_path, model_path, embedding_path, song_name, channel_name)

    # Step 2: Load metadata
    metadata1 = pd.read_csv(metadata_path)
    metadata2 = pd.read_csv(store_metadata_path_file)
    metadata_df = pd.concat([metadata1, metadata2], ignore_index=True)
    metadata_df["YT Link"] = "https://www.youtube.com/watch?v=" + metadata_df["video_id"]

    # Step 3: Load embeddings
    df = build_dataframe(song_embeddings, new_song_embeddings)

    # Step 4: Merge metadata
    df["query_key"] = (df["Artist"] + " " + df["Song"]).apply(normalize_text)
    metadata_df["meta_key"] = (metadata_df["channel_name"] + " " + metadata_df["song_title"]).apply(normalize_text)
    df = fuzzy_merge(df, metadata_df, "query_key", "meta_key", threshold=85)

    # Step 5: Cluster embeddings
    embedding_cols = [col for col in df.columns if col.startswith("e")]
    df[embedding_cols] = df[embedding_cols].astype('float32')
    embeddings = df[embedding_cols].values

    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    df = df.copy()  # defragment
    df[f"Cluster_{k}"] = cluster_labels

    # Step 6: Prepare query with normalized text
    normalized_song_name = normalize_text(song_name)
    normalized_channel_name = normalize_text(channel_name)
    
    # Try to find the song with normalized text comparison
    query_row = df[
        df.apply(
            lambda x: (normalize_text(x["Artist"]) == normalized_channel_name) & 
                     (normalize_text(x["Song"]) == normalized_song_name),
            axis=1
        )
    ]

    if query_row.empty:
        # If still not found, try fuzzy matching
        all_songs = df.apply(lambda x: f"{x['Artist']} {x['Song']}", axis=1).tolist()
        query_text = f"{channel_name} {song_name}"
        best_match = process.extractOne(
            normalize_text(query_text),
            [normalize_text(s) for s in all_songs],
            scorer=fuzz.token_sort_ratio,
            score_cutoff=85
        )
        
        if best_match:
            idx = all_songs.index(all_songs[best_match[2]])
            query_row = df.iloc[[idx]]
        else:
            raise ValueError(f"Query song '{song_name}' by '{channel_name}' not found in the DataFrame.")

    query_embedding = query_row[embedding_cols].values[0].astype('float32')

    # Step 7: FAISS within the same cluster
    query_cluster = query_row[f"Cluster_{k}"].values[0]
    cluster_df = df[df[f"Cluster_{k}"] == query_cluster].copy()
    cluster_embeddings = cluster_df[embedding_cols].values.astype('float32')
    
    # Increase n to ensure we have enough recommendations after filtering
    buffer_n = n + 1  # Add 1 to account for the query song
    indices, distances = find_closest_embeddings_faiss(cluster_embeddings, query_embedding, top_k=buffer_n)

    recommendations = cluster_df.iloc[indices].copy()
    recommendations["Distance"] = distances

    # Filter out query song
    recommendations = recommendations[
        ~((recommendations["Artist"] == channel_name) & (recommendations["Song"] == song_name))
    ]

    # If we don't have enough recommendations from the same cluster, get more from other clusters
    while len(recommendations) < n:
        # Find the next closest cluster
        remaining_clusters = sorted(set(df[f"Cluster_{k}"].unique()) - {query_cluster})
        if not remaining_clusters:
            break
            
        next_cluster = remaining_clusters[0]
        query_cluster = next_cluster  # Update for next iteration
        
        # Get recommendations from the next cluster
        next_cluster_df = df[df[f"Cluster_{k}"] == next_cluster].copy()
        next_cluster_embeddings = next_cluster_df[embedding_cols].values.astype('float32')
        next_indices, next_distances = find_closest_embeddings_faiss(
            next_cluster_embeddings, 
            query_embedding, 
            top_k=n - len(recommendations)
        )
        
        # Add new recommendations
        next_recommendations = next_cluster_df.iloc[next_indices].copy()
        next_recommendations["Distance"] = next_distances
        recommendations = pd.concat([recommendations, next_recommendations])

    # Ensure we have exactly n recommendations by taking the top n
    recommendations = recommendations.head(n)

    # Step 8: Ensure required columns exist
    required_columns = ["Artist", "Song", "Distance", "YT Link"]
    for col in required_columns:
        if col not in recommendations.columns:
            recommendations[col] = "N/A"

    df_plot = df.copy()
    # Filter the DataFrame to include only the query song and the recommended songs
    query_artist, query_song = channel_name, song_name
    recommended_songs = recommendations[["Artist", "Song"]]

    # Create a mask for the query song
    query_mask = (df["Artist"] == query_artist) & (df["Song"] == query_song)

    # Create a mask for the recommended songs
    recommended_mask = df.apply(lambda row: f"{row['Artist']}::{row['Song']}" in 
                                [f"{a}::{s}" for a, s in zip(recommended_songs["Artist"], recommended_songs["Song"])], axis=1)

    # Combine the masks
    combined_mask = query_mask | recommended_mask

    # Filter the DataFrame
    df_plot = df[combined_mask].copy()

    # ✅ Apply PCA to get 2D coordinates for visualization
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(df_plot[embedding_cols])
    df_plot['x'] = coords_2d[:, 0]
    df_plot['y'] = coords_2d[:, 1]

    # Step 9: Plot recommendations
    plot_recommendations(df_plot, query_id_creator(channel_name, song_name), recommendations, k=k, output_path=store_svg)
    
    # ✅ Prepare coordinates data for interface
    coordinates = []
    for _, row in df_plot.iterrows():
        coordinates.append({
            'artist': row['Artist'],
            'song': row['Song'],
            'x': float(row['x']),
            'y': float(row['y']),
            'is_query': (row['Artist'] == query_artist and row['Song'] == query_song)
        })
    
    cluster_col = f"Cluster_{k}"
    if cluster_col in recommendations.columns:
        recommendations = recommendations.drop(columns=[cluster_col])
    
    # ✅ Return both recommendations and coordinates
    return {
        'recommendations': recommendations,
        'coordinates': coordinates
    }



if __name__ == "__main__":
    # --- Command-line argument parsing ---
    parser = argparse.ArgumentParser(description="Run the Recommendator V4 script.")
    parser.add_argument("song_url", help="YouTube video URL of the song to process")
    args = parser.parse_args()

    # --- PARAMETERS TO EDIT ---
    song_path = "./recommendator/app/store_song"  # Folder for downloaded songs
    embedding_path = "./recommendator/app/store_embedding"  # Folder for generated embeddings
    model_path = "./recommendator/app/MODEL/model.pb"  # Pretrained model file
    metadata_path = "./recommendator/app/METADATA/metadata.csv"  # Metadata file
    store_metadata_path = "./recommendator/app/store_metadata"  # Folder for storing metadata
    song_embeddings = "./recommendator/app/EMBEDDINGS/embeddings.pkl"  # Embedding file
    store_svg = "./recommendator/app/SVG/plot.svg"

    metric = "euclidean"  # "cosine" or "euclidean"
    k = 1
    n = 5
    # --- END PARAMETERS ---

    # --- Run the recommendator ---
    top_recs = recomendator_with_faiss(
        song_path, args.song_url, embedding_path, model_path, metadata_path, store_metadata_path, song_embeddings, store_svg, k, n
    )
    print("Top recommendations:", top_recs)

