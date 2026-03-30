import json
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec


def init_pinecone(api_key: str, index_name: str):
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)


def ensure_index(api_key: str, index_name: str, dimension: int):
    pc = Pinecone(api_key=api_key)
    existing = {item["name"] for item in pc.list_indexes()}
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)


def load_nodes(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
