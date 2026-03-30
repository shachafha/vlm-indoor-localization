import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from RAG.embedding import load_embedding_model
from RAG.distance import estimate_distance
from RAG.frequency import build_frequency_table
from RAG.pinecone_index import init_pinecone
from RAG.runners import run_openai


def softmax_aggregate(scores: list[float]) -> float:
    """Log-sum-exp aggregation used for heuristic node weighting."""
    if not scores:
        return float("-inf")
    anchor = max(scores)
    return float(math.log(sum(math.exp(score - anchor) for score in scores) + 1e-9))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_query_items(query_file: Path) -> list[dict[str, Any]]:
    with query_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _serialize_candidates(
    top_nodes_sorted: list[tuple[str, float]],
    node_metadata_dict: dict[str, dict[str, Any]],
) -> str:
    payload = []
    for node_id, score in top_nodes_sorted:
        metadata = node_metadata_dict[node_id]
        payload.append(
            {
                "node_id": node_id,
                "location": metadata["location"],
                "direction": metadata["direction"],
                "score": round(score, 6),
                "filename": metadata.get("filename", ""),
            }
        )
    return json.dumps(payload, ensure_ascii=False)


def run_localization(config: dict[str, Any], api_keys: dict[str, str]) -> Path:
    floor_path = Path(config["floor_folder"])
    node_metadata_file = floor_path / config["node_metadata_file"]
    relative_file = floor_path / config["relative_locations_file"]
    query_file = floor_path / config["query_file"]

    output_csv = Path(config["output_csv"])
    topk = int(config["topk"])
    runs_per_query = int(config.get("runs_per_query", 1))
    retriever_model_tag = config["description_model_tag"]
    include_distance = bool(config.get("include_distance", False))

    _ensure_parent(output_csv)

    frequency_table, full_node_metadata = build_frequency_table(str(node_metadata_file))
    node_metadata_dict = {
        f"{item['location']}_{item['direction']}": item for item in full_node_metadata
    }

    with relative_file.open("r", encoding="utf-8") as handle:
        relative_text = handle.read().strip()

    query_items = _load_query_items(query_file)
    embedding_model = load_embedding_model(config["embedding_model"])
    index = init_pinecone(api_keys["PINECONE_API_KEY"], config["index_name"])

    fieldnames = [
        "query_image_name",
        "floor_name",
        "try_num",
        "identified_location",
        "facing_direction",
        "reasoning",
        "full_model_answer",
        "top_candidates",
        "predicted_distance_m",
        "distance_reason",
        "distance_relative_location",
        "distance_confidence",
        "distance_key_cues",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for query_index, query in enumerate(query_items):
            query_name = query["filename"]
            query_text = json.dumps(query["description"], ensure_ascii=False)
            print(f"[{query_index + 1}/{len(query_items)}] {query_name}")

            node_score_counter_raw: dict[str, list[float]] = defaultdict(list)
            visual_features = query["description"].get("visual_feature", [])
            for visual_feature in visual_features:
                description = visual_feature.get("Long form open description")
                if not description:
                    continue

                vector = embedding_model.encode(description)
                results = index.query(
                    vector=vector.tolist(),
                    top_k=topk,
                    include_metadata=True,
                    filter={"vlm_model": retriever_model_tag},
                )

                frequency = frequency_table.get(visual_feature.get("type", ""), 1)
                weight = 1 / (frequency + 1)
                for match in results.matches:
                    node_id = match.id.split("__")[0]
                    node_score_counter_raw[node_id].append(match.score * weight)

            node_score_counter = {
                node_id: softmax_aggregate(scores)
                for node_id, scores in node_score_counter_raw.items()
            }
            top_nodes_sorted = sorted(
                node_score_counter.items(), key=lambda item: item[1], reverse=True
            )[:topk]
            serialized_candidates = _serialize_candidates(top_nodes_sorted, node_metadata_dict)

            for attempt in range(runs_per_query):
                raw_reply = ""
                parsed: dict[str, Any]
                distance_result = {
                    "distance_m": "",
                    "distance_reason": "",
                    "distance_relative_location": "",
                    "distance_confidence": "",
                    "distance_key_cues": [],
                }
                try:
                    raw_reply = run_openai(
                        api_key=api_keys["OPENAI_API_KEY"],
                        node_score_counter=node_score_counter,
                        node_metadata_dict=node_metadata_dict,
                        relative_text=relative_text,
                        query_text=query_text,
                        topk=topk,
                        model_name=config["reasoning_model"],
                    )
                    start = raw_reply.find("{")
                    end = raw_reply.rfind("}")
                    if start == -1 or end == -1 or end <= start:
                        raise ValueError("Model response did not contain JSON.")
                    parsed = json.loads(raw_reply[start : end + 1])
                except Exception as exc:
                    parsed = {
                        "identified_location": "",
                        "facing_direction": "",
                        "reasoning": "",
                        "full_answer": raw_reply,
                    }
                    print(f"Failed on {query_name} try {attempt}: {exc}")

                if include_distance:
                    predicted_location = parsed.get("identified_location", "")
                    predicted_direction = str(parsed.get("facing_direction", "")).lower()
                    node_key = f"{predicted_location}_{predicted_direction}"
                    node_match = node_metadata_dict.get(node_key)
                    if node_match:
                        try:
                            distance_result = estimate_distance(
                                api_key=api_keys["OPENAI_API_KEY"],
                                node_desc=node_match["description"],
                                query_desc=query["description"],
                                model_name=config["reasoning_model"],
                            )
                        except Exception as exc:
                            print(f"Distance estimation failed on {query_name} try {attempt}: {exc}")

                writer.writerow(
                    {
                        "query_image_name": query_name,
                        "floor_name": config["floor_name"],
                        "try_num": attempt,
                        "identified_location": parsed.get("identified_location", ""),
                        "facing_direction": parsed.get("facing_direction", ""),
                        "reasoning": parsed.get("reasoning", ""),
                        "full_model_answer": parsed.get("full_answer", raw_reply),
                        "top_candidates": serialized_candidates,
                        "predicted_distance_m": distance_result.get("distance_m", ""),
                        "distance_reason": distance_result.get("distance_reason", ""),
                        "distance_relative_location": distance_result.get(
                            "distance_relative_location", ""
                        ),
                        "distance_confidence": distance_result.get("distance_confidence", ""),
                        "distance_key_cues": json.dumps(
                            distance_result.get("distance_key_cues", []),
                            ensure_ascii=False,
                        ),
                    }
                )

                print(
                    "  -> "
                    f"predicted {parsed.get('identified_location', '')} "
                    f"{parsed.get('facing_direction', '')}"
                    + (
                        f", distance {distance_result.get('distance_m', '')}m"
                        if include_distance and distance_result.get("distance_m", "") != ""
                        else ""
                    )
                )

    return output_csv
