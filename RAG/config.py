from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_run_config(
    config_path: str | Path,
    floor_name: str,
    query_set: str | None = None,
    output_csv: str | None = None,
    include_distance: bool = False,
) -> dict[str, Any]:
    raw = load_yaml_config(config_path)
    experiment = raw["experiment"]
    floors = raw["floors"]

    if floor_name not in floors:
        available = ", ".join(sorted(floors))
        raise KeyError(f"Unknown floor '{floor_name}'. Available floors: {available}")

    floor_cfg = floors[floor_name]
    selected_query_set = query_set or experiment["default_query_set"]
    if selected_query_set not in {"standard", "extra"}:
        raise ValueError("query_set must be 'standard' or 'extra'.")

    coordinate_files = floor_cfg.get("coordinate_files", {})
    query_file = floor_cfg["query_files"][selected_query_set]
    output_name = (
        output_csv
        or f"{floor_cfg['index_name']}_{experiment['reasoning_model']}_top{experiment['topk']}_{selected_query_set}.csv"
    )

    return {
        "floor_name": floor_name,
        "floor_folder": str(ROOT / floor_cfg["path"]),
        "index_name": floor_cfg["index_name"],
        "embedding_model": experiment["embedding_model"],
        "description_model": experiment["description_model"],
        "description_model_tag": experiment["description_model_tag"],
        "reasoning_model": experiment["reasoning_model"],
        "topk": experiment["topk"],
        "runs_per_query": experiment["runs_per_query"],
        "node_metadata_file": floor_cfg["node_descriptions"],
        "query_file": query_file,
        "relative_locations_file": floor_cfg["relative_locations_file"],
        "output_csv": str(ROOT / "outputs" / output_name),
        "images_locations_dir": str(ROOT / floor_cfg["images_locations_dir"]),
        "query_set": selected_query_set,
        "include_distance": include_distance,
        "meters_per_pixel": floor_cfg.get("meters_per_pixel"),
        "node_coordinates_file": coordinate_files.get("nodes", "nodes_locations_coordinates.csv"),
        "query_coordinates_file": coordinate_files.get(
            selected_query_set,
            "query_extra_locations_coordinates.csv"
            if selected_query_set == "extra"
            else "query_locations_coordinates.csv",
        ),
    }
