import os
import glob
import pickle
import argparse
import datetime
import numpy as np
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs

import csv


def compute_effnet_embeddings_for_folder(
    folder: str, model: str, metadata_csv: str, output_folder: str = None
) -> None:
    """
    Compute embeddings for every .mp3 in `folder` (including subdirectories) using metadata from a CSV.
    Writes to pickle with format `<foldername>-<date>-effnet.pkl`.
    Each entry contains `video_id`, `song_title`, `channel_name`, `youtube_link`, and `embedding`.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = model
    if output_folder is None:
        output_folder = base_dir
    os.makedirs(output_folder, exist_ok=True)

    # Load metadata CSV into dict
    metadata_map = {}
    with open(metadata_csv, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_id_key = row["video_id"]
            metadata_map[video_id_key] = (
                row["video_id"],
                row["song_title"],
                row["channel_name"],
                row["youtube_link"],
            )

    # Generate output filename based on folder name, current date, and model string
    date_str = datetime.date.today().isoformat()
    folder_basename = os.path.basename(os.path.normpath(folder))
    model_lower = model_path.lower()
    if "artist" in model_lower:
        suffix = "-artist"
    elif "multi" in model_lower:
        suffix = "-multi"
    else:
        suffix = ""
    output_filename = f"{folder_basename}-{date_str}-effnet{suffix}.pkl"
    output_path = os.path.join(output_folder, output_filename)

    # Initialize the embedding model
    model = TensorflowPredictEffnetDiscogs(
        graphFilename=model_path, output="PartitionedCall:1"
    )

    entries = []
    # Recursively process each MP3 file in sorted order
    for file_path in sorted(
        glob.glob(os.path.join(folder, "**", "*.mp3"), recursive=True)
    ):
        filename = os.path.basename(file_path)
        # Derive video_id and lookup metadata; skip if not found
        video_id = os.path.splitext(filename)[0]
        if video_id not in metadata_map:
            continue
        video_id, song_title, channel_name, youtube_link = metadata_map[video_id]

        # Load audio and compute embedding
        audio = MonoLoader(filename=file_path, sampleRate=16000, resampleQuality=4)()
        embedding = model(audio)
        embedding = np.mean(embedding, axis=0)  # Average over time
        # Convert embedding to a list of floats
        try:
            embedding_list = embedding.tolist()
        except AttributeError:
            embedding_list = list(embedding)

        entries.append(
            {
                "video_id": video_id,
                "song_title": song_title,
                "channel_name": channel_name,
                "youtube_link": youtube_link,
                "embedding": embedding_list,
            }
        )

    # Write all entries to pickle
    with open(output_path, "wb") as f:
        pickle.dump(entries, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Processed {len(entries)} files, saved embeddings to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute EffNet embeddings for all mp3s in a folder"
    )
    parser.add_argument("folder", help="Path to folder containing .mp3 files")
    parser.add_argument("model", help="Path to the TensorFlow Effnet model file")
    parser.add_argument("metadata_csv", help="Path to the metadata CSV file")
    parser.add_argument(
        "-o", "--output_folder", help="Directory to save pickle output", default=None
    )
    args = parser.parse_args()
    compute_effnet_embeddings_for_folder(
        args.folder, args.model, args.metadata_csv, args.output_folder
    )
