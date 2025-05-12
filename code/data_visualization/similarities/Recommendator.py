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

def load_maest_embeddings(path, population_label):
    with open(path, "rb") as f:
        data = pickle.load(f)
    rows = []
    for embedding_data, artist, song in data:
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


# --- Recommendation function ---
def recommend_similar_songs(df, query_id, n=5, k=12, metric='cosine'):
    assert metric in ['cosine', 'euclidean'], "Metric must be 'cosine' or 'euclidean'"
    assert f"Cluster_{k}" in df.columns, f"Cluster_{k} not found in DataFrame"

    try:
        artist, song = query_id.split("::")
    except ValueError:
        raise ValueError("query_id must be in the format 'Artist::Song'")

    query_row = df[(df["Artist"] == artist) & (df["Song"] == song)]
    if query_row.empty:
        raise ValueError(f"Song '{query_id}' not found in DataFrame")

    embedding_cols = [col for col in df.columns if col.startswith("e")]
    query_embedding = query_row[embedding_cols].values[0]
    query_cluster = query_row[f"Cluster_{k}"].values[0]

    cluster_df = df[df[f"Cluster_{k}"] == query_cluster].copy()
    cluster_df = cluster_df[~((cluster_df["Artist"] == artist) & (cluster_df["Song"] == song))]

    def similarity(row):
        candidate_embedding = row[embedding_cols].values
        return 1 - cosine(query_embedding, candidate_embedding) if metric == 'cosine' else euclidean(query_embedding, candidate_embedding)

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


# --- Choose model and load data ---
def build_dataframe(model="effnet"):
    if model == "effnet":
        before = load_effnet_embeddings("song_embeddings/embeddings_effnet/before_2012_effnet_embeddings.json", "Before 2012")
        after = load_effnet_embeddings("song_embeddings/embeddings_effnet/after_2018_effnet_embeddings.json", "After 2018")
    elif model == "maest":
        before = load_maest_embeddings("song_embeddings/Archivo_maest_pkl/before_2012_maest_embeddings.pkl", "Before 2012")
        after = load_maest_embeddings("song_embeddings/Archivo_maest_pkl/after_2018_maest_embeddings.pkl", "After 2018")
    else:
        raise ValueError("Model must be either 'effnet' or 'maest'")
    
    all_songs = before + after
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


if __name__ == "__main__":
    import pandas as pd

    # Load metadata
    metadata1 = pd.read_csv("../MTG-102/database_csv/db_after_2018.csv")
    metadata2 = pd.read_csv("../MTG-102/database_csv/db_before_2012.csv")
    metadata_df = pd.concat([metadata1, metadata2], ignore_index=True)
    metadata_df.columns = metadata_df.columns.str.strip()

    # Execution config
    model = "effnet"
    Song_name = "Jenifer"
    Artist_name = "Els_Catarres"
    k = 1
    n = 4
    metric = "euclidean"

    # Load embedding DataFrame
    df = build_dataframe(model=model)

    # --- DO NOT ALTER Artist/Song columns ---
    # Instead, create normalized keys for fuzzy matching
    df["query_key"] = (df["Artist"].str.replace("_", " ") + " " + df["Song"].str.replace("_", " ")).apply(normalize_text)
    metadata_df["meta_key"] = (metadata_df["Band"] + " " + metadata_df["Song Name"]).apply(normalize_text)

    # Fuzzy merge
    matched = fuzzy_merge(df, metadata_df[["meta_key", "YT Link"]], "query_key", "meta_key", threshold=85)

    # Keep original Artist/Song, YT Link is added
    df = matched  # overwrite with merged version

    # --- Generate query ID using original (underscored) names ---
    query_id = query_id_creator(Artist_name, Song_name)

    # KMeans clustering
    embedding_cols = [col for col in df.columns if col.startswith("e")]
    X = df[embedding_cols].values
    kmeans = KMeans(n_clusters=k, random_state=42)
    df[f"Cluster_{k}"] = kmeans.fit_predict(X)

    # Recommendations
    top_recs = recommend_similar_songs(df, query_id, n=n, k=k, metric=metric)
    print(top_recs)

    # t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X)
    df["x"] = X_2d[:, 0]
    df["y"] = X_2d[:, 1]
    df["Label"] = df["Artist"] + " - " + df["Song"]

    # Plot
    plot_recommendations(df, query_id, top_recs, k=k)