import json
from typing import List, Tuple

# Loader for datasets
def load_dataset(path: str) -> Tuple[List[str], List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [row["text"] for row in data]
    labels = [int(row["label"]) for row in data]

    return texts, labels