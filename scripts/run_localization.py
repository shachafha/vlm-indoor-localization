import argparse
import os
import sys

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from RAG.config import ROOT as PROJECT_ROOT, build_run_config
from RAG.pipeline import run_localization


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper localization pipeline.")
    parser.add_argument("--config", default="configs/paper_release.yaml")
    parser.add_argument("--floor", required=True)
    parser.add_argument("--output-name", default=None)
    parser.add_argument(
        "--include-distance",
        action="store_true",
        help="Also estimate query-to-predicted-node distance for each prediction.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY", ""),
    }
    for key, value in api_keys.items():
        if not value:
            raise RuntimeError(f"{key} is missing. Add it to .env first.")

    run_config = build_run_config(
        config_path=args.config,
        floor_name=args.floor,
        output_csv=args.output_name,
        include_distance=args.include_distance,
    )
    output_path = run_localization(run_config, api_keys)
    print(f"Predictions written to {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
