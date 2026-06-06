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
from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, Z_org_tensor, idx, true_labels):
    model.eval()
    with torch.no_grad():
        Z = Z_org_tensor[idx]
        logits = model.classify(Z)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    labels = [true_labels[i] for i in idx]
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1

class GIFTTrainer:
    def __init__(self, config):
        self.config = config

        self.word_graph_builder = WordGraphBuilder(glove_path=config["glove_path"])
        self.entity_graph_builder = EntityGraphBuilder(config["transe_path"], config["mapping_path"])
        self.pos_graph_builder = POSGraphBuilder()
        self.td_builder = TDMatrixBuilder()
        self.encoder = TextEncoder()
        self.svd = SVDViewGenerator(rank_ratio=config["rank_ratio"])

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

        return encode(word_graph), encode(entity_graph), encode(pos_graph)

    def build_td_matrices(self, texts, vocab, entities, pos_tags):
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
        kmeans = ConstrainedSeedKMeans(num_clusters=self.config["num_classes"])
        return kmeans.fit_predict(Z_org, labeled_idx, labels)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# def load_dataset(dataset_name):
#     split_path = f"data/original/{dataset_name}/{dataset_name.lower()}_split.json"
#     with open(split_path) as f:
#         data = json.load(f)
        
#     # DEBUG
#     print(data.keys())
#     for k, v in data.items():
#         print(k, len(v))

#     all_texts, all_labels = [], []
#     train_labeled_idx, val_idx, test_idx = [], [], []

#     for split_name, split_docs in data.items():
#         for i in sorted(split_docs.keys(), key=int):
#             doc = split_docs[i]
#             idx = len(all_texts)
#             all_texts.append(doc["text"])
#             all_labels.append(int(doc["label"]))

#             if split_name == "train":
#                 train_labeled_idx.append(idx)
#             elif split_name in ("val", "dev"):
#                 val_idx.append(idx)
#             else:
#                 test_idx.append(idx)

#     return {
#         "all_texts": all_texts,
#         "all_labels": all_labels,
#         "train_labeled_idx": train_labeled_idx,
#         "val_idx": val_idx,
#         "test_idx": test_idx,
#     }

def load_dataset(dataset_name):
    split_path = f"data/original/{dataset_name}/{dataset_name.lower()}_split.json"
    idx_path = f"data/original/{dataset_name}/{dataset_name.lower()}_idx_split.json"

    with open(split_path) as f:
        split_data = json.load(f)

    with open(idx_path) as f:
        idx_data = json.load(f)
        
    #  # DEBUG
    # print(data.keys())
    # for k, v in data.items():
    #     print(k, len(v))

    print("IDX KEYS:", idx_data.keys())
    print("IDX SAMPLE:", {k: v[:5] for k, v in idx_data.items()})

    all_texts, all_labels = [], []

    # Load train corpus (labeled + unlabeled)
    for i in sorted(split_data["train"].keys(), key=int):
        doc = split_data["train"][i]
        all_texts.append(doc["text"])
        all_labels.append(int(doc["label"]))

    train_corpus_size = len(all_texts)

    # Load test docs, appended after train
    for i in sorted(split_data["test"].keys(), key=int):
        doc = split_data["test"][i]
        all_texts.append(doc["text"])
        all_labels.append(int(doc["label"]))

    train_labeled_idx = idx_data["train"]
    val_idx = idx_data["valid"]
    test_idx = idx_data["test"] 

    return {
        "all_texts": all_texts,
        "all_labels": all_labels,
        "train_labeled_idx": train_labeled_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    # Load dataset
    data = load_dataset(args.dataset)
    all_texts = data["all_texts"]
    all_labels = data["all_labels"]
    train_labeled_idx = data["train_labeled_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    print(f"Total docs: {len(all_texts)}")
    print(f"Train labeled: {len(train_labeled_idx)}")
    print(f"Val: {len(val_idx)}")
    print(f"Test: {len(test_idx)}")
    print(f"Sum check: {len(train_labeled_idx) + len(val_idx) + len(test_idx)}")

    # Load preprocessed data
    base = f"data/processed/{args.dataset}"
    train_tokens = load_pickle(f"{base}/train_tokens.pkl")
    train_pos = load_pickle(f"{base}/train_pos.pkl")
    train_entities = load_pickle(f"{base}/train_entities.pkl")
    train_vocab = load_pickle(f"{base}/train_vocab.pkl")

    # Sanity checks
    assert not isinstance(train_entities[0], str), "entities are flattened"
    assert not isinstance(train_pos[0], str), "pos_tags are flattened"

    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

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

    # Build everything once before the training loop
    print("Building graphs...")
    word_graph, entity_graph, pos_graph = trainer.build_graphs(
        train_tokens, train_entities, train_pos
    )

    print("Running GCN...")
    H_w, H_e, H_p = trainer.run_gcn(word_graph, entity_graph, pos_graph)

    print("Building TD matrices...")
    M_w, M_e, M_p = trainer.build_td_matrices(
        all_texts, train_vocab, train_entities, train_pos
    )

    print("Running SVD...")
    M_wr, M_er, M_pr = trainer.run_svd(M_w, M_e, M_p)

    print("Building embeddings...")
    Z_org, Z_aug = trainer.build_embeddings(
        M_w, M_e, M_p, M_wr, M_er, M_pr, H_w, H_e, H_p
    )

    print("Running k-means...")
    weak_labels = trainer.run_kmeans(Z_org, train_labeled_idx, labels_tensor)

    # Convert to tensors once
    Z_org_tensor = torch.tensor(Z_org, dtype=torch.float32)
    Z_aug_tensor = torch.tensor(Z_aug, dtype=torch.float32)
    weak_labels_tensor = torch.tensor(weak_labels, dtype=torch.long)

    # Training loop
    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)

    metrics = {"train_loss": [], "val_acc": [], "val_f1": []}

    num_epochs = 10

    for epoch in range(num_epochs):
        trainer.model.train()
        optimizer.zero_grad()

        outputs = trainer.model(
            Z_org_tensor,
            Z_aug_tensor,
            weak_labels_tensor,
            train_labeled_idx, # Only training labeled indices
            labels_tensor
        )

        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        # Validate on val split only
        val_acc, val_f1 = evaluate(
            trainer.model,
            Z_org_tensor,
            val_idx,
            all_labels
        )

        metrics["train_loss"].append(loss.item())
        metrics["val_acc"].append(val_acc)
        metrics["val_f1"].append(val_f1)

        print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open(f"results/{args.dataset}_history.json", "w") as f:
        json.dump(metrics, f)

    os.makedirs("saved_models", exist_ok=True)
    torch.save(trainer.model.state_dict(), f"saved_models/{args.dataset}_gift.pt")
    print("Model saved.")