# --- STEP 1: Imports ---
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import plotly.express as px
import subprocess

# --- STEP 2: Load Embeddings ---
def load_embeddings(path, population_label):
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

before = load_embeddings("song_embeddings/before_2012_effnet_embeddings.json", "Before 2012")
after = load_embeddings("song_embeddings/after_2018_effnet_embeddings.json", "After 2018")
all_songs = before + after

# --- STEP 3: Flatten embeddings ---
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

# --- STEP 5: Static plot (unchanged) ---
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x="x", y="y", hue="Population", style="Artist")
plt.title("t-SNE of Essentia Audio Embeddings (Colored by Population)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

output_dir = "visualization_results/embedding_visualization_effnet"
os.makedirs(output_dir, exist_ok=True)
png_path = os.path.join(output_dir, "tsne_by_population.png")
plt.savefig(png_path, dpi=300)
plt.close()

# --- STEP 6: Interactive Plot with Custom Artist Shapes ---
artist_symbol_map = {
    "Antonia_Font": "circle",
    "Els_Catarres": "x",
    "Macedonia": "square",
    "Manel": "cross",
    "Marina_Rossell": "diamond",
    "Txarango": "triangle-up",
    "31_fam": "triangle-down",
    "julieta": "triangle-left",
    "la_ludwig_band": "triangle-right",
    "mushkaa": "star",
    "oques_grasses": "hexagon",
    "the_tyets": "pentagon"
}

df["Label"] = df["Artist"] + " - " + df["Song"]

fig = px.scatter(
    df,
    x="x", y="y",
    color="Population",
    symbol="Artist",  # <- assign shape by artist
    symbol_map=artist_symbol_map,  # <- assign fixed shapes
    hover_name="Label",
    title="Interactive t-SNE of Essentia Audio Embeddings",
    labels={"x": "t-SNE Dimension 1", "y": "t-SNE Dimension 2"},
    width=1000,
    height=750
)

# 4. Save and open
html_path = os.path.join(output_dir, "tsne_by_population_plotly.html")
fig.write_html(html_path)
print(f"✅ Interactive Plot saved to: {html_path}")

html_full_path = os.path.abspath(html_path)
if html_full_path.startswith("/mnt/"):
    drive_letter = html_full_path[5]
    windows_path = html_full_path.replace(f"/mnt/{drive_letter}/", f"{drive_letter.upper()}:\\").replace("/", "\\")
    subprocess.run(["powershell.exe", "Start-Process", windows_path])
else:
    print("⚠️ Could not convert path to Windows. Please open the HTML file manually.")
