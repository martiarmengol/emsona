# Paper we have read BERT.
# https://freedium.cfd/https://medium.com/data-science/masked-language-modelling-with-bert-7d49793e5d2c

# Step 1: Install HuggingFace Transformers if not installed
#!pip install transformers
#!pip install transformers scikit-learn matplotlib

# Step 2: Imports
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import plotly.express as px
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Step 3: Load multilingual model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModel.from_pretrained("xlm-roberta-base")

# Step 4: Function to compute a single embedding for full lyrics
def generate_lyrics_embedding(text, model, tokenizer, max_len=512, stride=256):
    # Tokenize input text
    tokens = tokenizer.encode(text, add_special_tokens=True)

    # If lyrics are short enough, no need to chunk
    if len(tokens) <= max_len:
        input_ids = torch.tensor(tokens).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS]-like token

        return cls_embedding.squeeze(0).cpu().numpy()

    # If too long, split into overlapping chunks
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_len, len(tokens))
        chunk = tokens[start:end]
        chunks.append(chunk)
        if end == len(tokens):
            break
        start += stride

    # Collect CLS embeddings for all chunks
    embeddings = []
    model.eval()
    with torch.no_grad():
        for chunk_ids in chunks:
            input_ids = torch.tensor(chunk_ids).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS]-like token
            embeddings.append(cls_embedding.squeeze(0))

    # Average all chunk embeddings to form one single embedding
    stacked = torch.stack(embeddings)
    avg_embedding = stacked.mean(dim=0).cpu().numpy()

    return avg_embedding

# Step 5: Example lyric (Catalan, long)
lyrics = """Amunt Si encara hi ets és un plaer Haver aguantat la vida amb tu Ens ha ajudat el fons On tot és fosc... (your full text here)"""

# Step 6: Get embedding
embedding = generate_lyrics_embedding(lyrics, model, tokenizer)

# Step 7: Show output
print(f"Generated embedding shape: {embedding.shape}")
print(embedding)

# ─── 1. Load your two datasets ─────────────────────────────────────────────────

paths = {
    "before_2012": "before_2012-2025-05-13-lyrics.pkl",
    "after_2018":  "after_2018-2025-05-13-lyrics.pkl",
}

data = {}
for label, p in paths.items():
    with open(p, "rb") as f:
        data[label] = pickle.load(f)
    print(f"{label}: {len(data[label])} songs")

# ─── 2. Prepare model + embedder ────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model     = AutoModel.from_pretrained("xlm-roberta-base")
model.eval()

def generate_lyrics_embedding(text, model, tokenizer,
                              max_len=512, stride=256):
    # tokenize into overflowing windows, *with padding to max_len*
    enc = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length",           # ← pad every chunk to 512 tokens
        stride=stride,
        return_overflowing_tokens=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"]       # (n_chunks, 512)
    attention_mask = enc["attention_mask"]  # (n_chunks, 512)

    # batch-forward all chunks
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    # grab the [CLS] embedding (first token) from each chunk
    cls_embs = out.last_hidden_state[:, 0, :]  # (n_chunks, hidden_dim)

    # average them into a single vector
    final_emb = cls_embs.mean(dim=0).cpu().numpy()
    return final_emb


# ─── 3. Embed everything ────────────────────────────────────────────────────────

all_embeddings = []
all_labels     = []
all_meta       = []  # (song, artist, label)

for label, items in data.items():
    for song in items:
        emb = generate_lyrics_embedding(song["lyrics"], model, tokenizer)
        all_embeddings.append(emb)
        all_labels.append(label)
        all_meta.append((song["song"], song["artist"], label))

X = np.vstack(all_embeddings)   # shape (N_total, 768)

# ─── 4. PCA → t-SNE ────────────────────────────────────────────────────────────

# 4.1 PCA to 50 dims
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X)

# 4.2 t-SNE to 2D
tsne = TSNE(
    n_components=2,
    perplexity=30,
    n_iter=1000,
    init="pca",
    random_state=42,
    verbose=1
)
X_2d = tsne.fit_transform(X_pca)

def interactive_tsne(df_meta, X_2d, color_by="artist"):
    df = (
        pd.DataFrame(X_2d, columns=["tsne_1","tsne_2"])
          .join(pd.DataFrame(df_meta, columns=["song","artist","era"]))
    )
    return px.scatter(
        df, x="tsne_1", y="tsne_2",
        color=color_by,
        hover_data=["song","era"],
        title=f"t-SNE colored by {color_by}"
    )

fig = interactive_tsne(all_meta, X_2d, color_by="artist")
fig.show()

df = pd.DataFrame(X_2d, columns=["tsne_1","tsne_2"])
meta_df = pd.DataFrame([{"song": song, "artist": artist, "era": era} for song, artist, era in all_meta])
df = pd.concat([df, meta_df], axis=1)
category_cols = [c for c in df.columns if c not in ["tsne_1","tsne_2","song"]]
for cat in category_cols:
    fig = px.scatter(
        df,
        x="tsne_1",
        y="tsne_2",
        color=cat,
        hover_data=["song"] + [c for c in category_cols if c != cat],
        title=f"t-SNE of Catalan Songs — colored by {cat.title()}"
    )
    fig.show()