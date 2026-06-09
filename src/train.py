import os
import numpy as np
import torch
import pickle
import json
import torch.nn.functional as F

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

def evaluate(model, Z_org, idx, true_labels):
    model.eval()
    with torch.no_grad():
        Z = Z_org[idx]
        logits = model.classify(Z)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = [true_labels[i] for i in idx]
    assert len(Z_org) == len(true_labels)
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1


class GIFTTrainer:
    def __init__(self, config):
        self.config = config

        self.gcn_w = None
        self.gcn_e = None
        self.gcn_p = None

        self.word_graph_builder = WordGraphBuilder(glove_path=config["glove_path"])
        self.entity_graph_builder = EntityGraphBuilder(config["transe_path"], config["mapping_path"])
        self.pos_graph_builder = POSGraphBuilder()
        self.td_builder = TDMatrixBuilder()
        self.svd = SVDViewGenerator(k=config["svd_k"])

        self.model = GIFTModel(
            input_dim=config["input_dim"],
            num_classes=config["num_classes"],
            hidden_dim=config["hidden_dim"],
            projection_dim=config["projection_dim"],
            temperature=config["temp"],
            eta=config["eta"],
            zeta=config["zeta"],
            batch_size=config["batch_size"],
        )

    def build_graphs(self, corpus_tokens, corpus_entities, corpus_pos_tags):
        return (
            self.word_graph_builder.build(corpus_tokens),
            self.entity_graph_builder.build(corpus_entities),
            self.pos_graph_builder.build(corpus_pos_tags),
        )

    def init_gcn(self, word_graph, entity_graph, pos_graph):
        # Creates GCN models and stores them on trainer
        self.gcn_w = GCNEncoder(
            input_dim=word_graph.X.shape[1],
            hidden_dim=self.config["hidden_dim"]
        )
        self.gcn_e = GCNEncoder(
            input_dim=entity_graph.X.shape[1],
            hidden_dim=self.config["hidden_dim"]
        )
        self.gcn_p = GCNEncoder(
            input_dim=pos_graph.X.shape[1],
            hidden_dim=self.config["hidden_dim"]
        )

    def run_gcn(self, X_w, A_w, X_e, A_e, X_p, A_p):
        # Runs GCN forward pass
        H_w = self.gcn_w(X_w, A_w)
        H_e = self.gcn_e(X_e, A_e)
        H_p = self.gcn_p(X_p, A_p)
        return H_w, H_e, H_p

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

    def build_embeddings(self, M_w_t, M_e_t, M_p_t, M_wr_t, M_er_t, M_pr_t, H_w, H_e, H_p):
        Z_w = M_w_t @ H_w
        Z_e = M_e_t @ H_e
        Z_p = M_p_t @ H_p
        Z_org = torch.cat([Z_w, Z_e, Z_p], dim=1)

        Z_wr = M_wr_t @ H_w
        Z_er = M_er_t @ H_e
        Z_pr = M_pr_t @ H_p
        Z_aug = torch.cat([Z_wr, Z_er, Z_pr], dim=1)

        return Z_org, Z_aug

    def run_kmeans(self, Z_org_numpy, labeled_idx, labels):
        kmeans = ConstrainedSeedKMeans(num_clusters=self.config["num_classes"])
        return kmeans.fit_predict(Z_org_numpy, labeled_idx, labels)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_dataset(dataset_name):
    # Load datasets
    split_path = f"data/original/{dataset_name}/{dataset_name.lower()}_split.json"
    idx_path = f"data/original/{dataset_name}/{dataset_name.lower()}_idx_split.json"

    with open(split_path) as f:
        split_data = json.load(f)

    with open(idx_path) as f:
        idx_data = json.load(f)

    print("IDX KEYS:", idx_data.keys())
    print("IDX SAMPLE:", {k: v[:5] for k, v in idx_data.items()})

    all_texts, all_labels = [], []

    for i in sorted(split_data["train"].keys(), key=int):
        doc = split_data["train"][i]
        all_texts.append(doc["text"])
        all_labels.append(int(doc["label"]))

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
    parser.add_argument("--num_classes", type=int, default=2)
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

    # Load preprocessed data
    base = f"data/processed/{args.dataset}"
    train_tokens = load_pickle(f"{base}/train_tokens.pkl")
    train_pos = load_pickle(f"{base}/train_pos.pkl")
    train_entities = load_pickle(f"{base}/train_entities.pkl")
    train_vocab = load_pickle(f"{base}/train_vocab.pkl")

    assert not isinstance(train_entities[0], str), "entities are flattened"
    assert not isinstance(train_pos[0], str), "pos_tags are flattened"

    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    print(f"Label distribution in train: {torch.bincount(labels_tensor[torch.tensor(train_labeled_idx)])}")
    print(f"Label distribution overall:  {torch.bincount(labels_tensor)}")
    
    class_counts = torch.bincount(
        labels_tensor[torch.tensor(train_labeled_idx)]
    ).float()

    weights = 1.0 / class_counts
    weights = weights / weights.sum()
    print("Class weights:", weights)

    config = {
        "glove_path": "data/external/glove/glove.pkl",
        "transe_path": "data/external/NELL_KG/transe.pkl",
        "mapping_path": "data/external/NELL_KG/entity_map.pkl",
        "svd_k": 15,
        "temp": 0.5,
        "input_dim": 384,
        "num_classes": args.num_classes,
        "hidden_dim": 128,
        "projection_dim": 128,
        "eta": 0.5,
        "zeta": 0.5,
        "batch_size": 256,
    }
    
    trainer = GIFTTrainer(config)

    # Build graphs
    print("Building graphs...")
    word_graph, entity_graph, pos_graph = trainer.build_graphs(
        train_tokens, train_entities, train_pos
    )

    # Init GCN models (weights shared across epochs)
    print("Initializing GCN...")
    trainer.init_gcn(word_graph, entity_graph, pos_graph)

    # Convert graph inputs to tensors
    X_w = torch.tensor(word_graph.X, dtype=torch.float32)
    A_w = torch.tensor(word_graph.A, dtype=torch.float32)
    X_e = torch.tensor(entity_graph.X, dtype=torch.float32)
    A_e = torch.tensor(entity_graph.A, dtype=torch.float32)
    X_p = torch.tensor(pos_graph.X, dtype=torch.float32)
    A_p = torch.tensor(pos_graph.A, dtype=torch.float32)

    # Build TD matrices
    print("Building TD matrices...")
    M_w, M_e, M_p = trainer.build_td_matrices(
        all_texts, train_vocab, train_entities, train_pos
    )

    # Run SVD
    print("Running SVD...")
    M_wr, M_er, M_pr = trainer.run_svd(M_w, M_e, M_p)
    
    # SVD views
    M_wr_t = torch.tensor(M_wr, dtype=torch.float32)
    M_er_t = torch.tensor(M_er, dtype=torch.float32)
    M_pr_t = torch.tensor(M_pr, dtype=torch.float32)

    # Convert TD matrices to tensors
    M_w_t = torch.tensor(M_w,  dtype=torch.float32)
    M_e_t = torch.tensor(M_e,  dtype=torch.float32)
    M_p_t = torch.tensor(M_p,  dtype=torch.float32)
    M_wr_t = torch.tensor(M_wr, dtype=torch.float32)
    M_er_t = torch.tensor(M_er, dtype=torch.float32)
    M_pr_t = torch.tensor(M_pr, dtype=torch.float32)

    # Run k-means once using initial GCN embeddings
    print("Building initial embeddings for k-means...")
    with torch.no_grad():
        H_w_init, H_e_init, H_p_init = trainer.run_gcn(X_w, A_w, X_e, A_e, X_p, A_p)
        Z_org_init, _ = trainer.build_embeddings(
            M_w_t, M_e_t, M_p_t, M_wr_t, M_er_t, M_pr_t,
            H_w_init, H_e_init, H_p_init
        )

    print("Running k-means...")
    train_labels_only = labels_tensor[train_labeled_idx]
    weak_labels = trainer.run_kmeans(
        Z_org_init.cpu().numpy(),
        train_labeled_idx,
        train_labels_only
    )
    
    weak_labels_tensor = torch.full(
        (len(weak_labels),),
        -1,
        dtype=torch.long
    )

    weak_labels_np = np.array(weak_labels)
    cluster_counts = np.bincount(weak_labels_np[weak_labels_np >= 0])

    for i, label in enumerate(weak_labels):
        if label >= 0 and cluster_counts[label] < 0.9 * len(weak_labels):
            weak_labels_tensor[i] = label

    # Preserve true labels
    for idx in train_labeled_idx:
        weak_labels_tensor[idx] = labels_tensor[idx]

    # Optimizer covers all trainable parameters
    optimizer = torch.optim.Adam(
        list(trainer.model.parameters()) +
        list(trainer.gcn_w.parameters()) +
        list(trainer.gcn_e.parameters()) +
        list(trainer.gcn_p.parameters()),
        lr=1e-3 # Learning rate 0.001
    )

    metrics = {"train_loss": [], "val_acc": [], "val_f1": []}
    print("Labeled sample classes:", labels_tensor[torch.tensor(train_labeled_idx)])
    
    num_epochs = 50

    for epoch in range(num_epochs):
        # Training step
        trainer.model.train()
        trainer.gcn_w.train()
        trainer.gcn_e.train()
        trainer.gcn_p.train()
        optimizer.zero_grad()

        # Recompute H and Z each epoch (build fresh computation graph)
        H_w, H_e, H_p = trainer.run_gcn(X_w, A_w, X_e, A_e, X_p, A_p)
        Z_org, Z_aug = trainer.build_embeddings(
            M_w_t, M_e_t, M_p_t, M_wr_t, M_er_t, M_pr_t,
            H_w, H_e, H_p
        )
        
        Z_org.shape[1] == config["input_dim"]
        print("Z_org shape:", Z_org.shape)
        
        # Run k-means every epoch on current Z_org
        with torch.no_grad():
            weak_labels = trainer.run_kmeans(
                Z_org.detach().cpu().numpy(),
                train_labeled_idx,
                labels_tensor[train_labeled_idx]
            )
        weak_labels_tensor = torch.tensor(weak_labels, dtype=torch.long)
        
        # Allow clusters to envolve with embedddings
        # if epoch % 10 == 0:
        #     with torch.no_grad():
        #         (weak_labels = trainer.run_kmeans
        #             Z_org.detach().cpu().numpy(),
        #             train_labeled_idx,
        #             labels_tensor[train_labeled_idx]
        #         )

        #     weak_labels_tensor = torch.full(
        #         (len(weak_labels),),
        #         -1,
        #         dtype=torch.long
        #     )

        #     weak_labels_np = np.array(weak_labels)
        #     cluster_counts = np.bincount(weak_labels_np[weak_labels_np >= 0])

        #     for i, label in enumerate(weak_labels):
        #         if label >= 0 and cluster_counts[label] < 0.9 * len(weak_labels):
        #             weak_labels_tensor[i] = label

        #     # Preserve labels
        #     for idx in train_labeled_idx:
        #         weak_labels_tensor[idx] = labels_tensor[idx]

        #     print("Refreshed k-means labels")

        outputs = trainer.model(
            Z_org,
            Z_aug,
            weak_labels_tensor,
            train_labeled_idx,
            labels_tensor
        )

        loss = outputs["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(trainer.model.parameters()) +
            list(trainer.gcn_w.parameters()) +
            list(trainer.gcn_e.parameters()) +
            list(trainer.gcn_p.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        # Validation step
        trainer.model.eval()
        trainer.gcn_w.eval()
        trainer.gcn_e.eval()
        trainer.gcn_p.eval()

        with torch.no_grad():
            H_w_eval, H_e_eval, H_p_eval = trainer.run_gcn(X_w, A_w, X_e, A_e, X_p, A_p)
            Z_org_eval, _ = trainer.build_embeddings(
                M_w_t, M_e_t, M_p_t, M_wr_t, M_er_t, M_pr_t,
                H_w_eval, H_e_eval, H_p_eval
            )

        val_acc, val_f1 = evaluate(trainer.model, Z_org_eval, val_idx, all_labels)

        metrics["train_loss"].append(loss.item())
        metrics["val_acc"].append(val_acc)
        metrics["val_f1"].append(val_f1)

        print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open(f"results/{args.dataset}_history.json", "w") as f:
        json.dump(metrics, f)

    os.makedirs("saved_models", exist_ok=True)
    torch.save({
        "model": trainer.model.state_dict(),
        "gcn_w": trainer.gcn_w.state_dict(),
        "gcn_e": trainer.gcn_e.state_dict(),
        "gcn_p": trainer.gcn_p.state_dict(),
    }, f"saved_models/{args.dataset}_gift.pt")
    print("Model saved.")