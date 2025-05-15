# --- STEP 1: Imports ---
import json
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import subprocess
from plotly.io import write_image

# --- STEP 2: Load Embeddings ---
def load_embeddings(path, population_label):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    rows = []
    for entry in data:
        # Obtener ID del embedding
        entry_id = entry["id"]
        artist = entry["artist"]
        song = entry["song"]
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

before = load_embeddings("../../essentia-models/effnet/embeddings/before_2012-2025-05-13-effnet-artist.pkl", "Before 2012")
after = load_embeddings("../../essentia-models/effnet/embeddings/after_2018-2025-05-13-effnet-artist.pkl", "After 2018")
all_songs = before + after

# --- STEP 3: Load metadata from CSVs ---
before_meta = pd.read_csv("../../../db_download/db_csv/before_2012.csv")
after_meta = pd.read_csv("../../../db_download/db_csv/after_2018.csv")

# Asegurar que 'ID' esté como columna en los metadatos
if 'ID' not in before_meta.columns:
    before_meta.rename(columns={'id': 'ID'}, inplace=True)
if 'ID' not in after_meta.columns:
    after_meta.rename(columns={'id': 'ID'}, inplace=True)

# Renombrar columnas para que coincidan con el formato esperado
if 'Band' in before_meta.columns and 'Artist' not in before_meta.columns:
    before_meta.rename(columns={'Band': 'Artist'}, inplace=True)
if 'Song Name' in before_meta.columns and 'Song' not in before_meta.columns:
    before_meta.rename(columns={'Song Name': 'Song'}, inplace=True)

if 'Band' in after_meta.columns and 'Artist' not in after_meta.columns:
    after_meta.rename(columns={'Band': 'Artist'}, inplace=True)
if 'Song Name' in after_meta.columns and 'Song' not in after_meta.columns:
    after_meta.rename(columns={'Song Name': 'Song'}, inplace=True)

# --- STEP 4: Flatten embeddings ---
flat_data = []
for song in all_songs:
    row = {
        "ID": song["ID"],     # Añadir ID a los datos aplanados
        "Song": song["Song"],
        "Artist": song["Artist"],
        "Population": song["Population"]
    }
    
    # Asegurarse de que Embedding es un array antes de iterarlo
    embedding = song["Embedding"]
    if not isinstance(embedding, np.ndarray):
        embedding = np.array([embedding])
        
    for i, val in enumerate(embedding):
        row[f"e{i}"] = val
    flat_data.append(row)

df = pd.DataFrame(flat_data)

# --- STEP 5: Merge metadata with embeddings ---
# Combine metadata
before_meta['Population'] = 'Before 2012'
after_meta['Population'] = 'After 2018'
all_meta = pd.concat([before_meta, after_meta], ignore_index=True)

# Realizar la fusión basada en ID
df_with_meta = pd.merge(
    df, 
    all_meta[['ID', 'Instrumentation', 'Genre', 'Acoustic vs Electronic', 'Gender Voice', 'Bpm', 'Artist', 'Song']],
    on='ID',
    how='left',
    suffixes=('', '_original')
)

# Verificar que la fusión fue exitosa
missing_metadata = df_with_meta['Instrumentation'].isna().sum()
if missing_metadata > 0:
    print(f"⚠️ Advertencia: {missing_metadata} embeddings no tienen metadatos correspondientes.")
else:
    print(f"✅ Todos los embeddings ({len(df_with_meta)}) se fusionaron correctamente con sus metadatos.")

# --- STEP 6: Preprocesar las características para mejor visualización ---
# Convertir Instrumentation de valores numéricos a categorías
def categorize_instrumentation(value):
    value = int(value) if pd.notna(value) and value != '' else 0
    if value == 2:
        return "Instruments (2)"
    elif value == 3:
        return "Instruments (3)"
    elif value == 4:
        return "Instruments (4)"
    elif value == 5:
        return "Instruments (5)"
    else:
        return "Unknown"

# Aplicar la transformación
if 'Instrumentation' in df_with_meta.columns:
    df_with_meta['Instrumentation_Category'] = df_with_meta['Instrumentation'].apply(categorize_instrumentation)

# Definir colores para Population
population_colors = {
    "Before 2012": "#1f77b4",  # Azul
    "After 2018": "#ff7f0e"    # Naranja
}

# Simplificar nombres de género para evitar leyendas muy largas
if 'Genre' in df_with_meta.columns:
    # Función para extraer el género principal
    def simplify_genre(genre):
        if pd.isna(genre) or genre == '':
            return 'Unknown'
        return genre.strip()
    
    df_with_meta['Genre_Main'] = df_with_meta['Genre'].apply(simplify_genre)
    
    # Obtener todos los géneros únicos para usarlos en la leyenda
    genre_colors = {
        "Pop": "#1f77b4",         # Azul
        "Folk": "#ff7f0e",        # Naranja
        "Urbà": "#2ca02c",        # Verde base
        "Urbà/Pop": "#7fbc41",    # Verde claro
        "Urbà/Trap": "#4d9221",   # Verde oscuro
        "Urbà/Trap Català": "#276419", # Verde muy oscuro
        "Urbà/Reggaeton": "#c51b7d",   # Rosa
        "Urbà/Bossa Nova": "#de77ae",  # Rosa claro
        "Reggaeton": "#d62728",   # Rojo
        "Trap": "#9467bd",        # Púrpura
        "Trap Català": "#8c6d31", # Marrón
        "Bossa Nova": "#8c564b",  # Marrón oscuro
        "Unknown": "#7f7f7f"      # Gris
    }
    
    # Añadir colores para cualquier otro género que aparezca
    unique_genres = df_with_meta['Genre_Main'].unique()
    other_colors = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]  # Colores adicionales
    i = 0
    for genre in unique_genres:
        if genre not in genre_colors and pd.notna(genre):
            genre_colors[genre] = other_colors[i % len(other_colors)]
            i += 1

# --- STEP 7: Apply t-SNE ---
embedding_cols = [col for col in df_with_meta.columns if col.startswith("e")]
X = df_with_meta[embedding_cols].values
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X)
df_with_meta["x"] = X_2d[:, 0]
df_with_meta["y"] = X_2d[:, 1]

# --- STEP 8: Static plot (unchanged) ---
plt.figure(figsize=(10, 7))

# Crear un mapa de colores personalizado para matplotlib
population_palette = {pop: color for pop, color in population_colors.items()}

# Usar el mapa de colores personalizado en el gráfico
sns.scatterplot(
    data=df_with_meta, 
    x="x", y="y", 
    hue="Population", 
    style="Artist", 
    palette=population_palette
)

plt.title("t-SNE of Effnet Artist Audio Embeddings (Colored by Population)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

output_dir = "visualization_results/embedding_visualization_effnet_artist_by_feature"
os.makedirs(output_dir, exist_ok=True)
png_path = os.path.join(output_dir, "tsne_by_population.png")
plt.savefig(png_path, dpi=300)
plt.close()

# --- STEP 9: Create Interactive Plots with Different Colorings ---
artist_symbol_map = {
    "antonia_font": "circle",
    "els_catarres": "x",
    "macedonia": "square",
    "manel": "cross",
    "marina_rossell": "diamond",
    "txarango": "triangle-up",
    "31_fam": "triangle-down",
    "julieta": "triangle-left",
    "la_ludwig_band": "triangle-right",
    "mushka": "star",
    "mushkaa": "star",  # Añadimos ambas variantes
    "oques_grasses": "hexagon",
    "the_tyets": "pentagon"
}

df_with_meta["Label"] = df_with_meta.apply(
    lambda row: f"{row.get('Artist_original', row['Artist'])} - {row.get('Song_original', row['Song'])}", 
    axis=1
)

# Function to create and save interactive plot
def create_interactive_plot(color_by, title_suffix=None, show_legend=True, custom_labels=None):
    if title_suffix is None:
        title_suffix = color_by
        
    # Determinar qué columna usar para cada característica
    column_to_use = color_by
    if color_by == "Instrumentation" and "Instrumentation_Category" in df_with_meta.columns:
        column_to_use = "Instrumentation_Category"
    elif color_by == "Genre" and "Genre_Main" in df_with_meta.columns:
        column_to_use = "Genre_Main"

    # Limpiar espacios en todas las características categóricas excepto Population
    if color_by != "Population" and color_by != "Bpm":
        df_with_meta[column_to_use] = df_with_meta[column_to_use].str.strip()
        # Normalizar Gender Voice para que Male & Female y Female & Male sean iguales
        if color_by == "Gender Voice":
            df_with_meta[column_to_use] = df_with_meta[column_to_use].apply(
                lambda x: "Male & Female" if pd.notna(x) and "Male" in x and "Female" in x else x
            )
    
    # Configurar paleta de colores personalizada para características categóricas
    if color_by == "Population":
        color_discrete_map = population_colors
    elif color_by == "Gender Voice":
        color_discrete_map = {
            "Male": "#2271B2",
            "Female": "#D55E00",
            "Male & Female": "#009E73"
        }
    elif color_by == "Acoustic vs Electronic":
        color_discrete_map = {
            "Acoustic": "#009E73",
            "Electronic": "#D55E00",
            "Acoustic & Electronic": "#CC79A7"
        }
    elif color_by == "Instrumentation" or color_by == "Instrumentation_Category":
        color_discrete_map = {
            "Instruments (2)": "#E69F00",
            "Instruments (3)": "#56B4E9",
            "Instruments (4)": "#009E73",
            "Instruments (5)": "#F0E442",
            "Unknown": "#999999"
        }
    elif color_by == "Genre":
        # Usar los colores definidos para géneros
        color_discrete_map = genre_colors if 'genre_colors' in globals() else None
    else:
        color_discrete_map = None
    
    # Para BPM, que es numérico, usar una paleta continua
    if color_by == "Bpm":
        color_continuous_scale = "Viridis"
    else:
        color_continuous_scale = None
    
    # Crear la figura con configuración explícita para la leyenda
    fig = px.scatter(
        df_with_meta,
        x="x", y="y",
        color=column_to_use,
        symbol="Artist",  # Mantener símbolos para todos los gráficos
        symbol_map=artist_symbol_map,
        hover_name="Label",
        color_discrete_map=color_discrete_map,
        color_continuous_scale=color_continuous_scale,
        title=f"Interactive t-SNE of Effnet Artist Audio Embeddings (Colored by {title_suffix})",
        labels={
            "x": "t-SNE Dimension 1", 
            "y": "t-SNE Dimension 2", 
            column_to_use: custom_labels or title_suffix
        },
        width=1200,  # Aumentar el ancho para dejar espacio para la leyenda
        height=750
    )
    
    # Añadir información sobre el tipo de coincidencia (exacta o aproximada)
    if "Match_Type" in df_with_meta.columns:
        fig.update_traces(
            hovertemplate='<b>%{hovertext}</b><br>Match: %{customdata[0]}<extra></extra>',
            customdata=df_with_meta[['Match_Type']]
        )
    
    # Ajustar la configuración de la leyenda según el tipo de visualización
    if color_by == "Bpm":
        # Para BPM, configurar solo la barra de color y ocultar la leyenda de símbolos
        fig.update_layout(
            coloraxis=dict(
                colorbar=dict(
                    title=dict(text="BPM", font=dict(size=16)),
                    thickness=20,
                    len=0.7,
                    x=1.02,
                    y=0.5
                )
            ),
            margin=dict(r=150)  # Aumentar el margen derecho para la barra de color
        )
        
        # Ocultar solo la leyenda de símbolos de artistas, pero mantener las formas en el gráfico
        for trace in fig.data:
            if hasattr(trace, 'showlegend'):
                trace.showlegend = False
                
    else:
        # Para otras visualizaciones, configurar la leyenda normal
        fig.update_layout(
            showlegend=True,  # Forzar la visibilidad de la leyenda
            legend=dict(
                title=dict(text=custom_labels or title_suffix, font=dict(size=16)),
                font=dict(size=14),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=2,
                itemsizing='constant',
                itemwidth=30,
                # Posicionar la leyenda fuera del gráfico a la derecha
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,  # Posición a la derecha del gráfico
                orientation="v"
            ),
            # Ajustar los márgenes para asegurarse de que hay espacio para la leyenda
            margin=dict(r=150)  # Aumentar el margen derecho
        )
        
        # Si estamos visualizando una característica categórica, suprimimos todos 
        # los símbolos de artistas de la leyenda
        for trace in fig.data:
            if hasattr(trace, 'marker') and hasattr(trace.marker, 'symbol'):
                trace.showlegend = False
        
        # Creamos traces específicos solo para la leyenda con cada categoría
        if color_by == "Population":
            # Para Population, mostrar explícitamente Before 2012 y After 2018
            for population, color in population_colors.items():
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None],  # Sin datos, solo para leyenda
                        mode='markers',
                        marker=dict(size=10, color=color),
                        name=str(population),
                        showlegend=True,
                        legendgroup=str(population)
                    )
                )
        elif color_by == "Genre" and 'genre_colors' in globals():
            # Para géneros, usar los colores personalizados
            for genre, color in genre_colors.items():
                if genre in df_with_meta[column_to_use].values:
                    fig.add_trace(
                        go.Scatter(
                            x=[None], y=[None],  # Sin datos, solo para leyenda
                            mode='markers',
                            marker=dict(size=10, color=color),
                            name=str(genre),
                            showlegend=True,
                            legendgroup=str(genre)
                        )
                    )
        elif color_discrete_map:
            # Para otras categorías, crear un elemento de leyenda para cada una
            categories = sorted([cat for cat in df_with_meta[column_to_use].unique() if pd.notna(cat)])
            for category in categories:
                if str(category) in color_discrete_map:
                    fig.add_trace(
                        go.Scatter(
                            x=[None], y=[None],  # Sin datos, solo para leyenda
                            mode='markers',
                            marker=dict(size=10, color=color_discrete_map[str(category)]),
                            name=str(category),
                            showlegend=True,
                            legendgroup=str(category)
                        )
                    )
    
    # Guardar el HTML interactivo
    html_path = os.path.join(output_dir, f"tsne_by_{color_by.lower().replace(' ', '_')}.html")
    fig.write_html(html_path)
    print(f"✅ Interactive Plot colored by {color_by} saved to: {html_path}")
    
    # Configuración para PNG: aumentar el tamaño para mejor calidad y asegurar espacio para leyendas
    fig_png = fig.update_layout(
        width=1600,         # Ancho mayor para la imagen
        height=1000,        # Alto mayor para la imagen
        margin=dict(r=200), # Margen derecho ampliado para asegurar que la leyenda sea visible
        font=dict(size=16), # Fuente más grande para mejor legibilidad en el PNG
    )
    
    # Generate PNG capture con alta resolución
    png_path = os.path.join(output_dir, f"tsne_by_{color_by.lower().replace(' ', '_')}.png")
    write_image(fig_png, png_path, scale=2)  # Escala 2x para mayor resolución
    print(f"✅ PNG capture saved to: {png_path}")
    
    return html_path

# Create plots for each feature
features = [
    {"name": "Population", "show_legend": True, "label": "Population"},
    {"name": "Instrumentation", "show_legend": True, "label": "Instrumentation Level"},
    {"name": "Genre", "show_legend": True, "label": "Music Genre"},
    {"name": "Acoustic vs Electronic", "show_legend": True, "label": "Production Type"},
    {"name": "Gender Voice", "show_legend": True, "label": "Gender Voice"},
    {"name": "Bpm", "show_legend": True, "label": "Beats per minute (BPM)"}
]

html_paths = []

for feature in features:
    if feature["name"] in df_with_meta.columns or (
        feature["name"] == "Instrumentation" and "Instrumentation_Category" in df_with_meta.columns
    ) or (
        feature["name"] == "Genre" and "Genre_Main" in df_with_meta.columns
    ):
        html_paths.append(create_interactive_plot(
            feature["name"],
            show_legend=feature["show_legend"],
            custom_labels=feature["label"]
        ))
    else:
        print(f"⚠️ Feature '{feature['name']}' no está disponible en los datos")

# Open the population plot (as in the original script)
if html_paths:
    html_full_path = os.path.abspath(html_paths[0])  # Population plot
    if html_full_path.startswith("/mnt/"):
        drive_letter = html_full_path[5]
        windows_path = html_full_path.replace(f"/mnt/{drive_letter}/", f"{drive_letter.upper()}:\\").replace("/", "\\")
        subprocess.run(["powershell.exe", "Start-Process", windows_path])
    else:
        print("⚠️ Could not convert path to Windows. Please open the HTML file manually.")


else:
    print("⚠️ No se crearon visualizaciones") 