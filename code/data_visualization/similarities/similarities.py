import os
import json
import pickle
import numpy as np
from scipy.spatial.distance import cosine, euclidean

EFFNET_DIR = os.path.join(os.path.dirname(__file__), "..", "song_embeddings", "embeddings_effnet")
MAEST_DIR = os.path.join(os.path.dirname(__file__), "..", "song_embeddings", "Archivo_maest_pkl")

def load_effnet_embeddings():
    embeddings = {}
    for fname in ["before_2012_effnet_embeddings.json", "after_2018_effnet_embeddings.json"]:
        with open(os.path.join(EFFNET_DIR, fname), "r") as f:
            data = json.load(f)
            for entry in data:
                key = f"{entry['artist']}::{entry['song']}"
                embedding = np.mean(entry["embedding"], axis=0)
                embeddings[key] = embedding
    return embeddings

def load_maest_embeddings():
    embeddings = {}
    for fname in ["before_2012_maest_embeddings.pkl", "after_2018_maest_embeddings.pkl"]:
        with open(os.path.join(MAEST_DIR, fname), "rb") as f:
            data = pickle.load(f)
            for embedding_data, artist, song in data:
                flattened = []
                for i in range(min(len(embedding_data), 100)):
                    if embedding_data[i]:
                        flattened.extend(embedding_data[i])
                flattened = np.array(flattened[:1000])
                key = f"{artist}::{song}"
                embeddings[key] = flattened
    return embeddings

def compute_similarity(embedding1, embedding2):
    cos_sim = 1 - cosine(embedding1, embedding2)
    euc_dist = euclidean(embedding1, embedding2)
    return cos_sim, euc_dist

# Load both models
effnet_embeddings = load_effnet_embeddings()
maest_embeddings = load_maest_embeddings()

# Example songs
song1_key = "oques_grasses::la_gent_que_estimo"
song2_key = "oques_grasses::inevitable"

if song1_key in effnet_embeddings and song2_key in effnet_embeddings:
    print("\n[EffNet Embeddings]")
    cos_sim, euc_dist = compute_similarity(effnet_embeddings[song1_key], effnet_embeddings[song2_key])
    print(f"Cosine Similarity: {cos_sim:.4f}")
    print(f"Euclidean Distance: {euc_dist:.4f}")
else:
    print("One or both EffNet songs not found.")

if song1_key in maest_embeddings and song2_key in maest_embeddings:
    print("\n[Maest Embeddings]")
    cos_sim, euc_dist = compute_similarity(maest_embeddings[song1_key], maest_embeddings[song2_key])
    print(f"Cosine Similarity: {cos_sim:.4f}")
    print(f"Euclidean Distance: {euc_dist:.4f}")
else:
    print("One or both Maest songs not found.")
