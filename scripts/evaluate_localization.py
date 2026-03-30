import argparse
import csv
from pathlib import Path
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from RAG.config import build_run_config
from RAG.evaluation import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate localization predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", default="configs/paper_release.yaml")
    parser.add_argument("--floor", default=None)
    parser.add_argument(
        "--evaluate-distance",
        action="store_true",
        help="Evaluate predicted distances when floor metadata is available.",
    )
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_path.parent / "evaluation"
    distance_eval_config = None
    if args.evaluate_distance:
        floor_name = args.floor
        if not floor_name and predictions_path.exists():
            with predictions_path.open("r", encoding="utf-8", newline="") as handle:
                first_row = next(iter(csv.DictReader(handle)), None)
            if first_row:
                floor_name = first_row.get("floor_name") or None

        if floor_name:
            run_config = build_run_config(
                config_path=args.config,
                floor_name=floor_name,
            )
            distance_eval_config = {
                "meters_per_pixel": run_config.get("meters_per_pixel"),
                "node_points_path": str(
                    Path(run_config["images_locations_dir"]) / run_config["node_coordinates_file"]
                ),
                "query_points_path": str(
                    Path(run_config["images_locations_dir"]) / run_config["query_coordinates_file"]
                ),
            }
        else:
            distance_eval_config = {"meters_per_pixel": None}

    detailed_path, summary_path = evaluate_predictions(
        predictions_path,
        output_dir,
        distance_eval_config=distance_eval_config,
    )
    print(f"Detailed evaluation saved to {detailed_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
