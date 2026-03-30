import base64
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def load_prompt(prompt_path: str | Path) -> str:
    with Path(prompt_path).open("r", encoding="utf-8") as handle:
        return handle.read().strip()


def list_images(folder: str | Path, allowed_filenames: set[str] | None = None) -> list[Path]:
    root = Path(folder)
    return sorted(
        path
        for path in root.iterdir()
        if path.suffix in IMAGE_EXTENSIONS
        and (allowed_filenames is None or path.name in allowed_filenames)
    )


def _encode_image(path: Path) -> str:
    with path.open("rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def _parse_json_response(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[len("```json") :]
    if text.startswith("```"):
        text = text[len("```") :]
    if text.endswith("```"):
        text = text[:-3]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Model response did not contain JSON.")
    return json.loads(text[start : end + 1].strip())


def _node_record_from_filename(filename: str) -> dict[str, Any]:
    match = re.fullmatch(r"node(\d+)_(north|south|east|west)\.[^.]+", filename, re.IGNORECASE)
    if not match:
        raise ValueError(f"Unsupported node filename format: {filename}")
    return {
        "filename": filename,
        "location": int(match.group(1)),
        "direction": match.group(2).lower(),
    }


def _query_record_from_filename(filename: str) -> dict[str, Any]:
    stem = Path(filename).stem.lower()
    numbers = [int(value) for value in re.findall(r"\d+", stem)]
    direction_match = re.search(r"(north|south|east|west)", stem)
    return {
        "filename": filename,
        "true_nodes": numbers,
        "true_direction": direction_match.group(1) if direction_match else None,
    }


def describe_images(
    api_key: str,
    images_folder: str | Path,
    prompt_path: str | Path,
    output_path: str | Path,
    model_name: str,
    image_kind: str,
    allowed_filenames: set[str] | None = None,
) -> Path:
    if image_kind not in {"node", "query"}:
        raise ValueError("image_kind must be 'node' or 'query'.")

    client = OpenAI(api_key=api_key)
    prompt = load_prompt(prompt_path)
    records = []

    for image_path in list_images(images_folder, allowed_filenames=allowed_filenames):
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        encoded = _encode_image(image_path)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{prompt}\n\nImage filename: {image_path.name}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{encoded}"},
                        },
                    ],
                }
            ],
        )
        raw_text = response.choices[0].message.content or ""
        description = _parse_json_response(raw_text)

        if image_kind == "node":
            record = _node_record_from_filename(image_path.name)
        else:
            record = _query_record_from_filename(image_path.name)

        record["description"] = description
        records.append(record)
        print(f"Processed {image_path.name}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)

    return output_path
