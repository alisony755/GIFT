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

def convert():
    print("Downloading AGNews CSV...")

    os.makedirs("data/agnews/raw", exist_ok=True)

    train_path = "data/original/agnews/raw/train.csv"
    test_path = "data/original/agnews/raw/test.csv"

    download(AGNEWS_URLS["train"], train_path)
    download(AGNEWS_URLS["test"], test_path)

    train = load_csv(train_path)
    test = load_csv(test_path)

    split_data = {
        "train": {str(i): x for i, x in enumerate(train)},
        "test": {str(i): x for i, x in enumerate(test)},
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