import os
import json
import urllib.request
import csv
from src.data.make_split import make_split

PROC_DIR = "data/original/agnews"
os.makedirs(PROC_DIR, exist_ok=True)

AGNEWS_URLS = {
    "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
}

def download(url, path):
    if not os.path.exists(path):
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)

def load_csv(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            label = int(row[0]) - 1
            text = row[1] + " " + row[2]
            data.append({"text": text, "label": label})
    return data

import random

def convert(max_train=4000, max_test=1000, seed=42):
    random.seed(seed)

    train_path = "data/original/agnews/raw/train.csv"
    test_path = "data/original/agnews/raw/test.csv"
    
    train = load_csv(train_path)
    test = load_csv(test_path)

    # Subsample while keeping class balance
    by_class = {}
    for item in train:
        by_class.setdefault(item["label"], []).append(item)
    
    per_class = max_train // 4
    subsampled_train = []
    for items in by_class.values():
        random.shuffle(items)
        subsampled_train.extend(items[:per_class])
    random.shuffle(subsampled_train)

    # Subsample test too
    random.shuffle(test)
    subsampled_test = test[:max_test]

    split_data = {
        "train": {str(i): x for i, x in enumerate(subsampled_train)},
        "test": {str(i): x for i, x in enumerate(subsampled_test)},
    }

    idx_split = make_split(split_data, num_classes=4)
 
    with open(os.path.join(PROC_DIR, "agnews_split.json"), "w") as f:
        json.dump(split_data, f)

    with open(os.path.join(PROC_DIR, "agnews_idx_split.json"), "w") as f:
        json.dump(idx_split, f)

    print("Saved dataset + splits")

    return split_data

if __name__ == "__main__":
    convert()