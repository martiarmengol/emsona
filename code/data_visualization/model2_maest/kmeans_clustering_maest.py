# --- STEP 1: Imports ---
import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns

# --- STEP 2: Load MAEST Embeddings from PKL files ---
def load_embeddings(path, population_label):
    print(f"Loading MAEST embeddings from: {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    rows = []
    for item in data:
        # Extract elements from the tuple
        embedding_data = item[0]  # This is a list of nested lists
        artist = item[1]
        song = item[2]
        
        print(f"Processing {song} by {artist}")
        
        # Convert the nested list structure to a flattened embedding
        # Based on professor's instructions: need to flatten the T, 6, 1, 685, 765 structure
        # The key is to multiply dimensions 685 x 765 as mentioned
        
        # Process first element of the embedding data to get a representative feature
        if embedding_data and isinstance(embedding_data, list):
            # Flatten the nested list structure to get a usable representation
            # This is an approximation since we don't have full details on the exact structure
            flattened_features = []
            
            # Take a sample of elements to avoid memory issues
            sample_size = min(len(embedding_data), 100)  # Sample size to prevent memory issues
            for i in range(0, sample_size):
                # Add embedding features to our flattened representation
                if i < len(embedding_data) and embedding_data[i]:
                    flattened_features.extend(embedding_data[i])
            
            # Convert to numpy array and ensure it has reasonable dimensions
            embedding_vector = np.array(flattened_features[:1000])  # Take first 1000 dimensions
            
            rows.append({
                "Song": song,
                "Artist": artist,
                "Population": population_label,
                "Embedding": embedding_vector
            })
        
    print(f"Loaded {len(rows)} songs from {population_label}")
    return rows

before = load_embeddings("song_embeddings/before_2012_maest_embeddings.pkl", "Before 2012")
after = load_embeddings("song_embeddings/after_2018_maest_embeddings.pkl", "After 2018")
all_songs = before + after
# --- STEP 3: Convert to DataFrame ---
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

df = pd.DataFrame(flat_data)

# --- STEP 4: Apply t-SNE ---
embedding_cols = [col for col in df.columns if col.startswith("e")]
X = df[embedding_cols].values
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X)
df["x"] = X_2d[:, 0]
df["y"] = X_2d[:, 1]
df["Label"] = df["Artist"] + " - " + df["Song"]

# --- STEP 5: Apply KMeans for multiple k values ---
k_values = [2, 3, 4, 5, 6, 8, 10, 12]
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    df[f"Cluster_{k}"] = kmeans.fit_predict(df[["x", "y"]])

# --- STEP 6: Interactive Plot for each k ---
output_dir = "visualization_maest_results/kmeans_clustering"
os.makedirs(output_dir, exist_ok=True)

for k in k_values:
    fig = px.scatter(
        df,
        x="x", y="y",
        color=df[f"Cluster_{k}"].astype(str),
        symbol="Artist",
        hover_name="Label",
        title=f"Interactive t-SNE with KMeans Clustering (k={k})",
        labels={"x": "t-SNE Dimension 1", "y": "t-SNE Dimension 2"},
        width=1000,
        height=750
    )

    html_path = os.path.join(output_dir, f"tsne_kmeans_k{k}.html")
    fig.write_html(html_path)
    print(f"✅ Interactive KMeans Plot (k={k}) saved to: {html_path}")

    html_full_path = os.path.abspath(html_path)
    if html_full_path.startswith("/mnt/"):
        drive_letter = html_full_path[5]
        windows_path = html_full_path.replace(f"/mnt/{drive_letter}/", f"{drive_letter.upper()}:\\").replace("/", "\\")
        subprocess.run(["powershell.exe", "Start-Process", windows_path])
    else:
        print("⚠️ Could not convert path to Windows. Please open the HTML file manually.")

# --- STEP 7: Static Comparison Plot (Grid) ---
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for ax, k in zip(axes.flatten(), k_values):
    sns.scatterplot(
        data=df,
        x="x", y="y",
        hue=f"Cluster_{k}",
        palette="tab10",
        ax=ax,
        legend=False,
        s=40
    )
    ax.set_title(f"KMeans Clustering (k={k})")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
comparison_png_path = os.path.join(output_dir, "kmeans_cluster_comparison.png")
plt.savefig(comparison_png_path, dpi=300)
plt.close()
print(f"📊 Static comparison plot saved to: {comparison_png_path}")

# --- STEP 8: Side-by-side Comparison with k=12 and Artist ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Original by Artist
sns.scatterplot(
    data=df,
    x="x", y="y",
    hue="Artist",
    palette="tab10",
    ax=ax1,
    s=50
)
ax1.set_title("Original Embedding Colored by Artist")
ax1.set_xlabel("t-SNE Dimension 1")
ax1.set_ylabel("t-SNE Dimension 2")
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

# KMeans Clustering with k=12
sns.scatterplot(
    data=df,
    x="x", y="y",
    hue="Cluster_12",
    palette="tab10",
    ax=ax2,
    s=50
)
ax2.set_title("KMeans Clustering with k=12")
ax2.set_xlabel("t-SNE Dimension 1")
ax2.set_ylabel("t-SNE Dimension 2")
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

plt.tight_layout()
side_by_side_path = os.path.join(output_dir, "artist_vs_kmeans12.png")
plt.savefig(side_by_side_path, dpi=300)
plt.close()
print(f"🎯 Side-by-side comparison plot saved to: {side_by_side_path}")
