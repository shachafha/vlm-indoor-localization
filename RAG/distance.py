import json
import math
import os
from pathlib import Path
from typing import Any

from openai import OpenAI


def build_distance_prompt(node_desc: dict[str, Any], query_desc: dict[str, Any]) -> str:
    return (
        "You are a visual localization expert. You will receive textual DESCRIPTIONS of "
        "two images of the same corridor taken from different camera positions.\n"
        "- NODE description = anchor view\n"
        "- QUERY description = current view\n\n"
        "Goal: estimate the physical forward/backward distance between the query camera "
        "and the node camera in meters. Use a signed value when possible, but if the "
        "sign is uncertain, still return your best numeric estimate.\n\n"
        "Important instructions:\n"
        "- Use only distinctive and reliable overlapping features.\n"
        "- Do not rely on repetitive or ambiguous features such as generic doors.\n"
        "- Leverage any `distance_from_camera_m` values in the descriptions.\n"
        "- Compare how far the same distinctive features appear in each description.\n"
        "- If no distinctive features overlap, make the best estimate you can and say so.\n\n"
        "NODE description:\n"
        f"{json.dumps(node_desc, ensure_ascii=False)}\n\n"
        "QUERY description:\n"
        f"{json.dumps(query_desc, ensure_ascii=False)}\n\n"
        "Return strict JSON only in this form:\n"
        "{\n"
        '  "distance_m": <number>,\n'
        '  "reason": "<short explanation>",\n'
        '  "relative_location": "<short phrase>",\n'
        '  "confidence": <0..1 number>,\n'
        '  "key_cues": ["cue1", "cue2"]\n'
        "}"
    )


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[len("```json") :]
    if text.startswith("```"):
        text = text[len("```") :]
    if text.endswith("```"):
        text = text[:-3]

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain JSON.")
    return json.loads(text[start : end + 1])


def estimate_distance(
    api_key: str,
    node_desc: dict[str, Any],
    query_desc: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": build_distance_prompt(node_desc, query_desc)}],
    )
    parsed = _extract_json_object(response.choices[0].message.content or "")

    try:
        distance_m = float(parsed.get("distance_m", parsed.get("distance", "")))
        if math.isnan(distance_m):
            distance_m = None
    except (TypeError, ValueError):
        distance_m = None

    confidence_raw = parsed.get("confidence")
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = None

    key_cues = parsed.get("key_cues", [])
    if not isinstance(key_cues, list):
        key_cues = [str(key_cues)]

    return {
        "distance_m": distance_m,
        "distance_reason": str(parsed.get("reason", "")),
        "distance_relative_location": str(
            parsed.get(
                "relative_location",
                parsed.get("relative location of query to node image", ""),
            )
        ),
        "distance_confidence": confidence,
        "distance_key_cues": [str(item) for item in key_cues],
    }


def normalize_coordinate_key(value: str) -> str:
    return os.path.splitext(value.strip().lower())[0]


def read_points_dict(path: str | Path) -> dict[str, tuple[float, float]]:
    import csv

    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return {}

    candidate_key_fields = (
        "location",
        "filename",
        "image",
        "image_name",
        "query_image_name",
        "node_image_name",
        "name",
    )
    key_field = next(
        (field for field in candidate_key_fields if field in rows[0]),
        reader.fieldnames[0] if reader.fieldnames else None,
    )
    if key_field is None:
        return {}

    points: dict[str, tuple[float, float]] = {}
    for row in rows:
        try:
            key = normalize_coordinate_key(str(row[key_field]))
            points[key] = (float(row["x"]), float(row["y"]))
        except (KeyError, TypeError, ValueError):
            continue
    return points
