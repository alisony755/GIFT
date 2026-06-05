import os
import pickle
import argparse
import json

from src.data.preprocess import Preprocessor
from src.data.build_vocab import VocabularyBuilder
from src.data.entity_extractor import EntityExtractor
from src.data.load_nell import load_ent2ids


def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def prepare_dataset(dataset_name):
    dataset_path = f"original_datasets/{dataset_name}/{dataset_name.lower()}_split.json"

    with open("data/external/NELL_KG/ent2ids_refined.pkl", "rb") as f:
        ent2ids = pickle.load(f)

    entity_extractor = EntityExtractor(ent2ids)
    preprocessor = Preprocessor()
    vocab_builder = VocabularyBuilder()

    with open(dataset_path, "r") as f:
        data = json.load(f)

    save_dir = f"processed/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)

    for split_name, split_data in data.items():

        texts = list(split_data["texts"])
        labels = list(split_data["labels"])

        tokens = preprocessor.tokenize(texts)
        pos_tags = preprocessor.pos_tag(tokens)

        vocab = vocab_builder.build(tokens)

        entities = entity_extractor.extract(texts)

        # DEBUG
        print(entities[:3])
        assert len(entities) == len(texts)

        save_pickle(f"{save_dir}/{split_name}_texts.pkl", texts)
        save_pickle(f"{save_dir}/{split_name}_tokens.pkl", tokens)
        save_pickle(f"{save_dir}/{split_name}_vocab.pkl", vocab)
        save_pickle(f"{save_dir}/{split_name}_entities.pkl", entities)
        save_pickle(f"{save_dir}/{split_name}_pos.pkl", pos_tags)
        save_pickle(f"{save_dir}/{split_name}_labels.pkl", labels)

        print(f"Saved {split_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    prepare_dataset(args.dataset)