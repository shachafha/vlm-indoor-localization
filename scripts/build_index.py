import argparse
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from RAG.config import build_run_config
from RAG.embedding import load_embedding_model
from RAG.pinecone_index import ensure_index, load_nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Pinecone index for a floor.")
    parser.add_argument("--config", default="configs/paper_release.yaml")
    parser.add_argument("--floor", required=True)
    args = parser.parse_args()

    load_dotenv()
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        raise RuntimeError("PINECONE_API_KEY is missing. Add it to .env first.")

    run_config = build_run_config(args.config, args.floor)
    model = load_embedding_model(run_config["embedding_model"])
    dimension = model.get_sentence_embedding_dimension()
    index = ensure_index(pinecone_key, run_config["index_name"], dimension)

    nodes = load_nodes(Path(run_config["floor_folder"]) / run_config["node_metadata_file"])
    for node in nodes:
        node_id = f"{node['location']}_{node['direction']}"
        visual_features = node["description"].get("visual_feature", [])
        for idx, feature in enumerate(visual_features):
            text = feature.get("Long form open description")
            if not text:
                continue
            vector = model.encode(text)
            metadata = {
                "location": node["location"],
                "direction": node["direction"],
                "filename": node["filename"],
                "vlm_model": run_config["description_model_tag"],
                "text": text,
            }
            index.upsert(
                vectors=[
                    {
                        "id": f"{node_id}__vf_{idx}__{run_config['description_model_tag']}",
                        "values": vector.tolist(),
                        "metadata": metadata,
                    }
                ]
            )

    print(f"Indexed {len(nodes)} node records into {run_config['index_name']}")


if __name__ == "__main__":
    main()
