import json

from openai import OpenAI


def run_openai(
    api_key,
    node_score_counter,
    node_metadata_dict,
    relative_text,
    query_text,
    topk,
    model_name,
):
    client = OpenAI(api_key=api_key)
    top_nodes_sorted = sorted(node_score_counter.items(), key=lambda item: item[1], reverse=True)[:topk]
    top_nodes = [node_metadata_dict[node_id] for node_id, _ in top_nodes_sorted]

    print(
        "Top nodes: "
        + ", ".join(f"{node['location']} {node['direction']}" for node in top_nodes)
    )

    template = (
        "You help blind and visually impaired people localize themselves on a building floor. "
        "You are given a query image description, relative floor relationships, and metadata for candidate nodes. "
        "Use the candidate evidence carefully and focus on stable, permanent visual cues.\n\n"
        "Relative locations:\n[RELATIVE]\n\n"
        "Query image description:\n[QUERY]\n\n"
        "Candidate node metadata will follow in separate messages.\n\n"
        "You may only answer with one of these candidates: [CHOICES]"
    )

    prompt = (
        template.replace("[RELATIVE]", relative_text)
        .replace("[QUERY]", query_text)
        .replace(
            "[CHOICES]",
            str([f"{node['location']} {node['direction']}" for node in top_nodes]),
        )
    )

    messages = [{"role": "user", "content": prompt}]
    for node in top_nodes:
        node_chunk = json.dumps(node, ensure_ascii=False, indent=2)
        messages.append({"role": "user", "content": f"Node metadata:\n{node_chunk}"})

    messages.append(
        {
            "role": "user",
            "content": (
                "Return strict JSON only:\n"
                "{\n"
                '  "identified_location": <numbers only>,\n'
                '  "facing_direction": <north|south|east|west>,\n'
                '  "reasoning": "<brief explanation>",\n'
                '  "full_answer": "<short restatement of the prediction>"\n'
                "}"
            ),
        }
    )

    response = client.chat.completions.create(model=model_name, messages=messages)
    if getattr(response, "usage", None):
        usage = response.usage
        print(
            "Tokens used: "
            f"prompt={usage.prompt_tokens}, "
            f"completion={usage.completion_tokens}, "
            f"total={usage.total_tokens}"
        )
    return response.choices[0].message.content
