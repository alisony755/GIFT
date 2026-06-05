import json

class DatasetLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        with open(self.path, "r") as f:
            data = json.load(f)

        train = data["train"]
        test = data["test"]

        train_texts = []
        train_labels = []

        for item in train.values():
            train_texts.append(item["text"])
            train_labels.append(item["label"])

        test_texts = []
        test_labels = []

        for item in test.values():
            test_texts.append(item["text"])
            test_labels.append(item["label"])

        return {
            "train_texts": train_texts,
            "train_labels": train_labels,
            "test_texts": test_texts,
            "test_labels": test_labels
        }