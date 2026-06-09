import random
from collections import defaultdict


def make_split(split_data, num_classes, n_per_class=20, seed=42):
    random.seed(seed)

    by_class = defaultdict(list)

    for i, item in enumerate(split_data["train"].values()):
        by_class[item["label"]].append(i)

    train_idx = []
    val_idx = []

    for c in range(num_classes):
        idxs = by_class[c]
        random.shuffle(idxs)

        k = min(len(idxs), n_per_class * 2)

        train_idx += idxs[:n_per_class]
        val_idx += idxs[n_per_class:k]

    test_idx = list(
        range(
            len(split_data["train"]),
            len(split_data["train"]) + len(split_data["test"])
        )
    )

    return {
        "train": train_idx,
        "valid": val_idx,
        "test": test_idx
    }