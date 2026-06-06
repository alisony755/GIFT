import os
import pickle
import argparse
import json

from src.data.preprocess import Preprocessor
from src.data.build_vocab import VocabularyBuilder
from src.data.entity_extractor import EntityExtractor
from src.data.load_nell import load_ent2ids

import time

def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# def prepare_dataset(dataset_name):
#     dataset_path = f"data/original/{dataset_name}/{dataset_name.lower()}_split.json"

#     with open("data/external/NELL_KG/ent2ids_refined.pkl", "rb") as f:
#         ent2ids = pickle.load(f)
        
#     print(list(ent2ids.keys())[:20])

#     preprocessor = Preprocessor()
#     entity_extractor = EntityExtractor(ent2ids)
#     vocab_builder = VocabularyBuilder()

#     with open(dataset_path, "r") as f:
#         data = json.load(f)

#     save_dir = f"data/processed/{dataset_name}"
#     os.makedirs(save_dir, exist_ok=True)

#     for split_name, split_data in data.items():

#         texts = []
#         labels = []

#         for i in sorted(split_data.keys(), key=int):
#             texts.append(split_data[i]["text"])
#             labels.append(int(split_data[i]["label"]))

#         tokens = preprocessor.tokenize(texts)
#         pos_tags = preprocessor.pos_tag(tokens)

#         vocab = vocab_builder.build(tokens)

#         start = time.time()
#         entities = entity_extractor.extract(texts)
#         print("Entity extraction time:", time.time() - start)

#         # DEBUG
#         print("NUM DOCS:", len(texts))
#         print("NUM ENTITY OUTPUTS:", len(entities))
#         print("SAMPLE:", entities[:3])
#         assert len(entities) == len(texts)

#         save_pickle(f"{save_dir}/{split_name}_texts.pkl", texts)
#         save_pickle(f"{save_dir}/{split_name}_tokens.pkl", tokens)
#         save_pickle(f"{save_dir}/{split_name}_vocab.pkl", vocab)
#         save_pickle(f"{save_dir}/{split_name}_entities.pkl", entities)
#         save_pickle(f"{save_dir}/{split_name}_pos.pkl", pos_tags)
#         save_pickle(f"{save_dir}/{split_name}_labels.pkl", labels)

#         print(f"Saved {split_name}")

def prepare_dataset(dataset_name):
    dataset_path = f"data/original/{dataset_name}/{dataset_name.lower()}_split.json"

    with open("data/external/NELL_KG/ent2ids_refined.pkl", "rb") as f:
        ent2ids = pickle.load(f)

    preprocessor = Preprocessor()
    entity_extractor = EntityExtractor(ent2ids)
    vocab_builder = VocabularyBuilder()

    with open(dataset_path, "r") as f:
        data = json.load(f)

    # Combine ALL splits into one corpus
    texts = []
    labels = []

    for split_data in data.values():

        for i in sorted(split_data.keys(), key=int):

            texts.append(split_data[i]["text"])
            labels.append(int(split_data[i]["label"]))

    print("TOTAL DOCS:", len(texts))

    tokens = preprocessor.tokenize(texts)

    pos_tags = preprocessor.pos_tag(tokens)

    vocab = vocab_builder.build(tokens)

    entities = entity_extractor.extract(texts)

    print("NUM DOCS:", len(texts))
    print("NUM ENTITY OUTPUTS:", len(entities))

    save_dir = f"data/processed/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)

    save_pickle(f"{save_dir}/train_texts.pkl", texts)
    save_pickle(f"{save_dir}/train_tokens.pkl", tokens)
    save_pickle(f"{save_dir}/train_vocab.pkl", vocab)
    save_pickle(f"{save_dir}/train_entities.pkl", entities)
    save_pickle(f"{save_dir}/train_pos.pkl", pos_tags)
    save_pickle(f"{save_dir}/train_labels.pkl", labels)

    print("Saved full dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    prepare_dataset(args.dataset)