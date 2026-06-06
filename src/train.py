import os
import numpy as np
import torch
import pickle
import json

from src.graph.word_graph_builder import WordGraphBuilder
from src.graph.entity_graph_builder import EntityGraphBuilder
from src.graph.pos_graph_builder import POSGraphBuilder

from src.text_representation.gcn_encoder import GCNEncoder
from src.text_representation.td_matrix_builder import TDMatrixBuilder
from src.text_representation.text_encoder import TextEncoder

from src.representation.svd_views import SVDViewGenerator

from src.contrastive.contrastive_loss import ContrastiveLoss
from src.contrastive.cluster_contrastive_loss import ClusterContrastiveLoss

from src.clustering.constrained_seed_kmeans import ConstrainedSeedKMeans

from src.model.gift_model import GIFTModel

class GIFTTrainer:
    def __init__(self, config):
        self.config = config

        self.word_graph_builder = WordGraphBuilder(
            glove_path=config["glove_path"]
        )

        self.entity_graph_builder = EntityGraphBuilder(
            config["transe_path"],
            config["mapping_path"]
        )

        self.pos_graph_builder = POSGraphBuilder()

        self.td_builder = TDMatrixBuilder()
        self.encoder = TextEncoder()

        self.svd = SVDViewGenerator(
            rank_ratio=config["rank_ratio"]
        )

        self.contrastive_loss = ContrastiveLoss(
            temperature=config["temp"]
        )

        self.cluster_loss = ClusterContrastiveLoss(
            temperature=config["temp"]
        )

        self.model = GIFTModel(
            input_dim=config["input_dim"],
            num_classes=config["num_classes"],
            hidden_dim=config["hidden_dim"],
            projection_dim=config["projection_dim"],
            temperature=config["temp"],
            eta=config["eta"],
            zeta=config["zeta"]
        )

    def build_graphs(self, corpus_tokens, corpus_entities, corpus_pos_tags):
        return (
            self.word_graph_builder.build(corpus_tokens),
            self.entity_graph_builder.build(corpus_entities),
            self.pos_graph_builder.build(corpus_pos_tags),
        )

    def run_gcn(self, word_graph, entity_graph, pos_graph):
        def encode(graph):
            model = GCNEncoder(
                input_dim=graph.X.shape[1],
                hidden_dim=self.config["hidden_dim"]
            )
            x = torch.tensor(graph.X, dtype=torch.float32)
            a = torch.tensor(graph.A, dtype=torch.float32)
            return model(x, a).detach().cpu().numpy()

        return (
            encode(word_graph),
            encode(entity_graph),
            encode(pos_graph),
        )

    def build_td_matrices(self, texts, vocab, entities, pos_tags):
        n_docs = len(texts)

        entities = entities[:n_docs]
        pos_tags = pos_tags[:n_docs]

        M_w = self.td_builder.build_word_matrix(texts, vocab)
        M_e = self.td_builder.build_entity_matrix(texts, entities)

        flat_pos_vocab = list(set(sum(pos_tags, [])))
        M_p = self.td_builder.build_pos_matrix(pos_tags, flat_pos_vocab)

        return M_w, M_e, M_p

    def run_svd(self, M_w, M_e, M_p):
        return (
            self.svd.build_augmented_matrix(M_w),
            self.svd.build_augmented_matrix(M_e),
            self.svd.build_augmented_matrix(M_p),
        )

    def build_embeddings(self, M_w, M_e, M_p, M_wr, M_er, M_pr, H_w, H_e, H_p):
        Z_org = self.encoder.build_original(M_w, M_e, M_p, H_w, H_e, H_p)
        Z_aug = self.encoder.build_augmented(M_wr, M_er, M_pr, H_w, H_e, H_p)
        return Z_org, Z_aug

    def run_kmeans(self, Z_org, labeled_idx, labels):
        kmeans = ConstrainedSeedKMeans(
            num_clusters=self.config["num_classes"]
        )

        return kmeans.fit_predict(Z_org, labeled_idx, labels)

    def forward(self, batch):
        texts = batch["texts"]
        tokens = batch["tokens"]
        vocab = batch["vocab"]
        entities = batch["entities"]
        pos_tags = batch["pos"]

        word_graph, entity_graph, pos_graph = self.build_graphs(
            tokens, entities, pos_tags
        )

        H_w, H_e, H_p = self.run_gcn(word_graph, entity_graph, pos_graph)
        
        M_w, M_e, M_p = self.build_td_matrices(
            texts, vocab, entities, pos_tags
        )
        
        # DEBUG
        print("M_w:", M_w.shape)
        print("M_e:", M_e.shape)
        print("M_p:", M_p.shape)

        M_wr, M_er, M_pr = self.run_svd(M_w, M_e, M_p)

        Z_org, Z_aug = self.build_embeddings(
            M_w, M_e, M_p,
            M_wr, M_er, M_pr,
            H_w, H_e, H_p
        )

        weak_labels = self.run_kmeans(
            Z_org,
            batch["labeled_idx"],
            batch["labels_tensor"]
        )

        return self.model(
            torch.tensor(Z_org, dtype=torch.float32),
            torch.tensor(Z_aug, dtype=torch.float32),
            torch.tensor(weak_labels, dtype=torch.long),
            batch["labeled_idx"],
            batch["labels_tensor"]
        )


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_dataset(dataset_name):
    split_path = f"data/original/{dataset_name}/{dataset_name.lower()}_split.json"
    idx_path = f"data/original/{dataset_name}/{dataset_name.lower()}_idx_split.json"

    with open(split_path) as f:
        data = json.load(f)

    with open(idx_path) as f:
        idx = json.load(f)

    all_docs = []

    for split in data.values():
        for i in sorted(split.keys(), key=int):
            all_docs.append(split[i])

    all_texts = [doc["text"] for doc in all_docs]
    all_labels = [int(doc["label"]) for doc in all_docs]

    train_idx = idx["train"]
    train_texts = [all_texts[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]

    return train_texts, train_labels, train_idx


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    train_texts, train_labels, train_idx = load_dataset(args.dataset)

    # Make labeled indices relative to filtered train set
    labeled_idx = list(range(len(train_texts)))

    base = f"data/processed/{args.dataset}"
    
    # DEBUG
    print("BASE PATH:", base)
    print("FILES:", os.listdir(base))

    train_tokens = load_pickle(f"{base}/train_tokens.pkl")
    train_pos = load_pickle(f"{base}/train_pos.pkl")
    train_entities = load_pickle(f"{base}/train_entities.pkl")
    train_vocab = load_pickle(f"{base}/train_vocab.pkl")

    labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    
    # DEBUG
    if len(train_entities) != len(train_texts):
        train_entities = train_entities[:len(train_texts)]

    if len(train_pos) != len(train_texts):
        train_pos = train_pos[:len(train_texts)]
    
    # DEBUG
    print("DEBUG BEFORE TRAINING")
    print("texts:", len(train_texts))
    print("entities:", len(train_entities))
    print("pos:", len(train_pos))
    
    print("ENTITY SAMPLE:", train_entities[:2])
    print("POS SAMPLE:", train_pos[:2])

    # HARD FAIL IF FLATTENED
    if isinstance(train_entities[0], str):
        raise ValueError("entities are flattened — extractor output is wrong")

    if isinstance(train_pos[0], str):
        raise ValueError("pos_tags are flattened — preprocess output is wrong")

    batch = {
        "texts": train_texts,
        "tokens": train_tokens,
        "entities": train_entities,
        "pos": train_pos,
        "vocab": train_vocab,
        "labels_tensor": labels_tensor,
        "labeled_idx": labeled_idx,
    }

    config = {
        "glove_path": "data/external/glove/glove.pkl",
        "transe_path": "data/external/NELL_KG/transe.pkl",
        "mapping_path": "data/external/NELL_KG/entity_map.pkl",
        "rank_ratio": 0.5,
        "temp": 0.5,
        "input_dim": 768,
        "num_classes": 2,
        "hidden_dim": 256,
        "projection_dim": 128,
        "eta": 0.5,
        "zeta": 0.5
    }

    trainer = GIFTTrainer(config)
    
    # DEBUG
    print("ENTITY TYPE CHECK:")
    print(type(train_entities))
    print("FIRST ITEM TYPE:", type(train_entities[0]))
    print("FIRST ITEM:", train_entities[0])
    
    print("FINAL CHECK:",
      len(train_texts),
      len(train_entities),
      len(train_pos))

    outputs = trainer.forward(batch)

    print(outputs["loss"].item())