# visualize.py
import os
import json
import argparse
import torch
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.train import load_dataset, load_pickle, GIFTTrainer


def plot_curves(dataset_name):
    with open(f"results/{dataset_name}_history.json") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{dataset_name} — Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{dataset_name}_loss.png")
    plt.close()
    print(f"Saved loss curve.")

    # Accuracy/F1 curve
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["val_acc"], label="Val Accuracy", color="green")
    plt.plot(epochs, history["val_f1"], label="Val Macro-F1", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{dataset_name} — Validation Accuracy & F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{dataset_name}_accuracy.png")
    plt.close()
    print(f"Saved accuracy curve.")


def plot_confusion_matrix(model, Z_org, test_idx, true_labels, dataset_name, class_names=None):
    model.eval()
    with torch.no_grad():
        Z = torch.tensor(Z_org[test_idx], dtype=torch.float32)
        logits = model.classify(Z)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    labels = [true_labels[i] for i in test_idx]

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    plt.title(f"{dataset_name} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"results/{dataset_name}_confusion.png")
    plt.close()
    print(f"Saved confusion matrix.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # Load data
    data = load_dataset(args.dataset)
    base = f"data/processed/{args.dataset}"

    tokens = load_pickle(f"{base}/train_tokens.pkl")
    pos = load_pickle(f"{base}/train_pos.pkl")
    entities = load_pickle(f"{base}/train_entities.pkl")
    vocab = load_pickle(f"{base}/train_vocab.pkl")

    config = {
        "glove_path": "data/external/glove/glove.pkl",
        "transe_path": "data/external/NELL_KG/transe.pkl",
        "mapping_path": "data/external/NELL_KG/entity_map.pkl",
        "svd_k": 15,
        "temp": 0.5,
        "input_dim": 768,
        "num_classes": 2,
        "hidden_dim": 256,
        "projection_dim": 128,
        "eta": 0.5,
        "zeta": 0.5,
        "batch_size": 256,
    }

    # Rebuild embeddings
    trainer = GIFTTrainer(config)

    word_graph, entity_graph, pos_graph = trainer.build_graphs(tokens, entities, pos)
    H_w, H_e, H_p = trainer.run_gcn(word_graph, entity_graph, pos_graph)
    M_w, M_e, M_p = trainer.build_td_matrices(
        data["all_texts"], vocab, entities, pos
    )
    Z_org, _ = trainer.build_embeddings(
        M_w, M_e, M_p, M_w, M_e, M_p,
        H_w, H_e, H_p
    )

    # Load saved weights
    trainer.model.load_state_dict(
        torch.load(f"saved_models/{args.dataset}_gift.pt")
    )
    trainer.model.eval()
    print("Loaded model weights.")

    # Plot loss + accuracy curves (from saved history)
    plot_curves(args.dataset)

    # Plot confusion matrix (on test set)
   
   # UPDATE CLASS NAMES PER DATASET
    class_names = ["negative", "positive"]  # for MR/Twitter

    plot_confusion_matrix(
        trainer.model,
        Z_org,
        data["test_idx"],
        data["all_labels"],
        dataset_name=args.dataset,
        class_names=class_names
    )

    print("All plots saved to results/")