# --- STEP 1: Imports ---
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import plotly.express as px
import subprocess

# --- STEP 2: Load MAEST Embeddings from PKL files ---
def load_embeddings(path, population_label, token_index=None):
    """
    Carga embeddings MAEST desde archivos PKL.
    
    Args:
        path: Ruta al archivo PKL.
        population_label: Etiqueta para la población (before/after).
        token_index: Índice del token de embeddings a utilizar.
                     Si es None, usa todos los tokens como antes.
                     Si es un número (0 o 1), usa solo ese token específico.
    """
    if token_index is None:
        print(f"Loading MAEST embeddings from: {path} (using ALL tokens)")
    else:
        print(f"Loading MAEST embeddings from: {path} (using token {token_index} only)")
        
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    rows = []
    for item in data:
        # Extract elements from the tuple
        embedding_data = item[0]  # This is a list of nested lists
        artist = item[1]
        song = item[2]
        
        print(f"Processing {song} by {artist}")
        
        # Dos modalidades: usar un token específico o todos los tokens
        if embedding_data and isinstance(embedding_data, list):
            if token_index is not None and token_index < len(embedding_data):
                # Modo 1: Usar solo el token específico
                selected_token = embedding_data[token_index]
                if isinstance(selected_token, list) and selected_token:
                    flattened_features = selected_token
            else:
                # Modo 2: Usar todos los tokens
                flattened_features = []
                sample_size = min(len(embedding_data), 100)
                for i in range(0, sample_size):
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

# Opciones para el uso de tokens:
# - None: usar todos los tokens del embedding
# - 0: usar solo el token 0 para reducir ruido
# - 1: usar solo el token 1 para reducir ruido
selected_token = 0  # Cambiar este valor o poner None para usar todos los tokens

# Usando el token seleccionado para los embeddings
before = load_embeddings("../../essentia-models/maest/embeddings/before_2012-2025-05-08_maest.pkl", "Before 2012", token_index=selected_token)
after = load_embeddings("../../essentia-models/maest/embeddings/after_2018-2025-05-08_maest.pkl", "After 2018", token_index=selected_token)
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

# Título dinámico según el modo seleccionado
if selected_token is None:
    title_suffix = "All Tokens"
    file_suffix = "all_tokens"
else:
    title_suffix = f"Token {selected_token} Only"
    file_suffix = f"token{selected_token}"

plt.title(f"t-SNE of MAEST Audio Embeddings ({title_suffix})")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

output_dir = "visualization_maest_results/embedding_visualization"
os.makedirs(output_dir, exist_ok=True)
png_path = os.path.join(output_dir, f"tsne_by_population_{file_suffix}.png")
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
    title=f"Interactive t-SNE of MAEST Audio Embeddings ({title_suffix})",
    labels={"x": "t-SNE Dimension 1", "y": "t-SNE Dimension 2"},
    width=1000,
    height=750
)

# 4. Save and open
html_path = os.path.join(output_dir, f"tsne_by_population_{file_suffix}_plotly.html")
fig.write_html(html_path)
print(f"✅ Interactive Plot saved to: {html_path}")

html_full_path = os.path.abspath(html_path)
if html_full_path.startswith("/mnt/"):
    drive_letter = html_full_path[5]
    windows_path = html_full_path.replace(f"/mnt/{drive_letter}/", f"{drive_letter.upper()}:\\").replace("/", "\\")
    subprocess.run(["powershell.exe", "Start-Process", windows_path])
else:
    print("⚠️ Could not convert path to Windows. Please open the HTML file manually.")
