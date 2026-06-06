import json

class DatasetLoader:
    def init(self, path):
        self.path = path

    def load(self):
        with open(self.path, "r") as f:
            data = json.load(f)

        return {
            "train": self._parse_split(data["train"]),
            "valid": self._parse_split(data["valid"]),
            "test": self._parse_split(data["test"])
        }

    def _parse_split(self, split):
        texts = []
        labels = []

        for item in split.values():
            texts.append(item["text"])
            labels.append(int(item["label"]))

        return {
            "texts": texts,
            "labels": labels
        }