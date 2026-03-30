import json
from collections import defaultdict
import os

def build_frequency_table(node_metadata_file):
    with open(node_metadata_file, "r", encoding="utf-8") as f:
        nodes = json.load(f)
    freq_table = defaultdict(int)
    for node in nodes:
        for vf in node.get("description", {}).get("visual_feature", []):
            freq_table[vf["type"]] += 1
    return freq_table, nodes
