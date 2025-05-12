# --- STEP 1: Imports ---
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- STEP 2: Load Dataset CSVs (metadata only) ---
before_path = "database_csv/db_before_2012.csv"
after_path = "database_csv/db_after_2018.csv"

before_df = pd.read_csv(before_path)
after_df = pd.read_csv(after_path)

before_df["Population"] = "Before 2012"
after_df["Population"] = "After 2018"

metadata_df = pd.concat([before_df, after_df], ignore_index=True)

# --- STEP 3: Preprocess Metadata ---
# Drop non-informative columns
data = metadata_df.drop(columns=["ID", "Song Name", "YT Link"])

# Encode categorical features using one-hot encoding
data_encoded = pd.get_dummies(data, columns=["Band", "Genre", "Acoustic vs Electronic", "Gender Voice"])

# Normalize BPM
data_encoded["Bpm"] = StandardScaler().fit_transform(data_encoded[["Bpm"]])

# Drop 'Population' column before PCA
if "Population" in data_encoded.columns:
    data_encoded = data_encoded.drop(columns=["Population"])

# --- ❗ DROP NaNs ---
data_encoded = data_encoded.dropna()

# --- STEP 4: Labels for Plotting ---
metadata_df["Label"] = metadata_df["Band"] + " - " + metadata_df["Song Name"]
labels = metadata_df["Label"].loc[data_encoded.index]
bands = metadata_df["Band"].loc[data_encoded.index]
populations = metadata_df["Population"].loc[data_encoded.index]

# --- STEP 5: Apply PCA ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(data_encoded)

# --- STEP 6: Create DataFrame for Plotting ---
pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Band": bands.values,
    "Population": populations.values,
    "Label": labels.values
})

# --- STEP 7: Plot and Save ---
plt.figure(figsize=(10, 7))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Band", style="Population")
plt.title("PCA of Metadata (without audio embeddings)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

output_dir = "visualization_results/pca_original_effnet"
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "pca_original_plot.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"✅ PCA metadata plot saved to: {plot_path}")

# --- STEP 8: Show explained variance and component importances ---
explained_variance = pca.explained_variance_ratio_
print("\n📊 Explained Variance Ratio:")
print(f"PC1: {explained_variance[0]:.4f}")
print(f"PC2: {explained_variance[1]:.4f}")

# --- Show top contributing features for each PC ---
components_df = pd.DataFrame(pca.components_, columns=data_encoded.columns, index=["PC1", "PC2"])

print("\n🔍 Top Feature Importances for PC1:")
print(components_df.loc["PC1"].abs().sort_values(ascending=False).head(10))

print("\n🔍 Top Feature Importances for PC2:")
print(components_df.loc["PC2"].abs().sort_values(ascending=False).head(10))
