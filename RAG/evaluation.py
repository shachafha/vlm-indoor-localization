import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from RAG.distance import normalize_coordinate_key, read_points_dict


def extract_numbers(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"(\d+)", text)]


def extract_first_number(text: str) -> int | None:
    values = extract_numbers(text)
    return values[0] if values else None


def extract_direction(text: str) -> str | None:
    lowered = text.lower()
    for direction in ("north", "south", "east", "west"):
        if direction in lowered:
            return direction
    return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _evaluate_distance_rows(
    rows: list[dict[str, str]],
    distance_eval_config: dict[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    meters_per_pixel = distance_eval_config.get("meters_per_pixel")
    if meters_per_pixel in (None, ""):
        return {}, {"distance_evaluation_skipped_reason": "meters_per_pixel is missing for this floor."}

    node_points_path = distance_eval_config["node_points_path"]
    query_points_path = distance_eval_config["query_points_path"]
    if not Path(node_points_path).exists() or not Path(query_points_path).exists():
        return {}, {
            "distance_evaluation_skipped_reason": (
                "Coordinate CSV files required for distance evaluation were not found."
            )
        }

    node_points = read_points_dict(node_points_path)
    query_points = read_points_dict(query_points_path)
    if not node_points or not query_points:
        return {}, {
            "distance_evaluation_skipped_reason": (
                "Coordinate CSV files could not be parsed into usable x/y mappings."
            )
        }

    per_row_metrics: dict[tuple[str, str], dict[str, Any]] = {}
    errors: list[float] = []
    within_half_meter = 0

    for row in rows:
        predicted_distance_raw = row.get("predicted_distance_m", "")
        try:
            predicted_distance = abs(float(predicted_distance_raw))
        except (TypeError, ValueError):
            continue

        identified_location = str(row.get("identified_location", "")).strip()
        facing_direction = str(row.get("facing_direction", "")).strip().lower()
        if not identified_location or not facing_direction:
            continue

        query_key = normalize_coordinate_key(row["query_image_name"])
        node_key = normalize_coordinate_key(f"node{identified_location}_{facing_direction}")
        query_point = query_points.get(query_key)
        node_point = node_points.get(node_key)
        if query_point is None or node_point is None:
            continue

        true_distance = math.dist(query_point, node_point) * float(meters_per_pixel)
        abs_error = abs(true_distance - predicted_distance)
        within = abs_error <= 0.5
        row_key = (row["query_image_name"], row["try_num"])
        per_row_metrics[row_key] = {
            "true_distance_m": round(true_distance, 6),
            "predicted_distance_m": round(predicted_distance, 6),
            "distance_abs_error_m": round(abs_error, 6),
            "distance_within_0_5m": within,
        }
        errors.append(abs_error)
        within_half_meter += int(within)

    if not errors:
        return {}, {
            "distance_evaluation_skipped_reason": (
                "No rows had both predicted distance values and matching coordinate metadata."
            )
        }

    mean_error = sum(errors) / len(errors)
    variance = sum((value - mean_error) ** 2 for value in errors) / len(errors)
    return per_row_metrics, {
        "num_distance_rows": len(errors),
        "distance_mean_abs_error_m": mean_error,
        "distance_std_abs_error_m": math.sqrt(variance),
        "distance_within_0_5m_rate": within_half_meter / len(errors),
    }


def evaluate_predictions(
    predictions_csv: str | Path,
    output_dir: str | Path,
    distance_eval_config: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    predictions_path = Path(predictions_csv)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(predictions_path)
    distance_row_metrics: dict[tuple[str, str], dict[str, Any]] = {}
    distance_summary: dict[str, Any] = {}
    if distance_eval_config is not None:
        distance_row_metrics, distance_summary = _evaluate_distance_rows(rows, distance_eval_config)

    detailed_rows = []
    correct_location = 0
    correct_total = 0

    for row in rows:
        actual_locations = extract_numbers(row["query_image_name"])
        actual_direction = extract_direction(row["query_image_name"])
        predicted_location = extract_first_number(row["identified_location"])
        predicted_direction = extract_direction(row["facing_direction"])

        location_hit = predicted_location in actual_locations if predicted_location is not None else False
        direction_hit = predicted_direction == actual_direction if location_hit else False
        total_hit = location_hit and direction_hit
        row_key = (row["query_image_name"], row["try_num"])
        distance_metrics = distance_row_metrics.get(row_key, {})

        correct_location += int(location_hit)
        correct_total += int(total_hit)
        detailed_rows.append(
            {
                "query_image_name": row["query_image_name"],
                "try_num": row["try_num"],
                "actual_locations": json.dumps(actual_locations),
                "actual_direction": actual_direction or "",
                "predicted_location": predicted_location if predicted_location is not None else "",
                "predicted_direction": predicted_direction or "",
                "location_correct": location_hit,
                "direction_correct_given_location": direction_hit,
                "total_correct": total_hit,
                "true_distance_m": distance_metrics.get("true_distance_m", ""),
                "predicted_distance_m": distance_metrics.get("predicted_distance_m", ""),
                "distance_abs_error_m": distance_metrics.get("distance_abs_error_m", ""),
                "distance_within_0_5m": distance_metrics.get("distance_within_0_5m", ""),
            }
        )

    total_rows = len(rows)
    location_accuracy = correct_location / total_rows if total_rows else math.nan
    direction_accuracy = correct_total / correct_location if correct_location else math.nan
    total_accuracy = correct_total / total_rows if total_rows else math.nan

    detailed_path = output_root / f"{predictions_path.stem}_detailed.csv"
    with detailed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed_rows[0].keys()) if detailed_rows else [])
        if detailed_rows:
            writer.writeheader()
            writer.writerows(detailed_rows)

    summary_path = output_root / f"{predictions_path.stem}_summary.json"
    summary = {
        "predictions_file": str(predictions_path),
        "num_rows": total_rows,
        "location_accuracy": location_accuracy,
        "direction_accuracy_given_location": direction_accuracy,
        "total_accuracy": total_accuracy,
    }
    summary.update(distance_summary)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return detailed_path, summary_path
