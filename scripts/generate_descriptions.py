import argparse
import os
import sys

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from RAG.config import ROOT as PROJECT_ROOT, load_yaml_config
from RAG.descriptions import describe_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GPT-5 mini image descriptions.")
    parser.add_argument("--config", default="configs/paper_release.yaml")
    parser.add_argument("--floor", required=True)
    parser.add_argument("--image-kind", choices=["node", "query"], required=True)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env first.")

    config = load_yaml_config(args.config)
    floor_cfg = config["floors"][args.floor]
    floor_root = PROJECT_ROOT / floor_cfg["path"]
    prompt_path = PROJECT_ROOT / "prompts" / "image_description_prompt.txt"
    model_name = config["experiment"]["description_model"]

    if args.image_kind == "node":
        images_folder = floor_root / floor_cfg["node_images_dir"]
        output_path = floor_root / floor_cfg["node_descriptions"]
        allowed_filenames = None
    else:
        images_folder = floor_root / floor_cfg["query_images_dir"]
        output_path = floor_root / floor_cfg["query_file"]
        allowed_filenames = None

    result_path = describe_images(
        api_key=api_key,
        images_folder=images_folder,
        prompt_path=prompt_path,
        output_path=output_path,
        model_name=model_name,
        image_kind=args.image_kind,
        allowed_filenames=allowed_filenames,
    )
    print(f"Saved descriptions to {result_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
