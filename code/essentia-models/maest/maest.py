import os
import glob
import pickle
import argparse
import datetime
from essentia.standard import MonoLoader, TensorflowPredictMAEST
import numpy as np


def compute_maest_embeddings_for_folder(folder: str, output_folder: str = None) -> None:
    """
    Compute embeddings for every .mp3 in `folder` (including subdirectories) and write to pickle.
    Output filename is current date (YYYY-MM-DD) with format `<date>_maest.pkl`.
    Each entry in the pickle contains `id`, `artist`, `song`, and `embedding`.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = "code/essentia-models/maest/discogs-maest-30s-pw-1.pb"
    if output_folder is None:
        output_folder = base_dir
    os.makedirs(output_folder, exist_ok=True)

    # Generate output filename based on current date
    date_str = datetime.date.today().isoformat()
    output_filename = f"{date_str}_maest.pkl"
    output_path = os.path.join(output_folder, output_filename)

    # Initialize the embedding model
    model = TensorflowPredictMAEST(
        graphFilename=model_path, output="StatefulPartitionedCall:7"
    )

    entries = []
    # Recursively process each MP3 file in sorted order
    for file_path in sorted(
        glob.glob(os.path.join(folder, "**", "*.mp3"), recursive=True)
    ):
        filename = os.path.basename(file_path)
        # Split into id, song, and artist based on filename structure <id>-<song>-<artist>
        name_no_ext = os.path.splitext(filename)[0]
        parts = name_no_ext.rsplit("-", 2)
        if len(parts) == 3:
            id_str, song_str, artist_str = parts
        elif len(parts) == 2:
            id_str, song_str = parts
            artist_str = "unknown"
        else:
            id_str = parts[0]
            song_str = ""
            artist_str = "unknown"

        # Load audio and compute embedding
        audio = MonoLoader(filename=file_path, sampleRate=16000, resampleQuality=4)()
        embedding = model(audio)
        # Convert embedding to a list of floats
        try:
            embedding_list = embedding.tolist()
        except AttributeError:
            embedding_list = list(embedding)

        # Reduce embedding dimensionality by averaging and squeezing
        arr = np.array(embedding_list)
        embedding_list = np.squeeze(arr.mean(axis=0), axis=0).tolist()

        entries.append(
            {
                "id": id_str,
                "song": song_str,
                "artist": artist_str,
                "embedding": embedding_list,
            }
        )

    # Write all entries to JSON
    with open(output_path, "wb") as f:
        pickle.dump(entries, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Processed {len(entries)} files, saved embeddings to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute MAEST embeddings for all mp3s in a folder"
    )
    parser.add_argument("folder", help="Path to folder containing .mp3 files")
    parser.add_argument(
        "-o", "--output_folder", help="Directory to save pickle output", default=None
    )
    args = parser.parse_args()
    compute_maest_embeddings_for_folder(args.folder, args.output_folder)
