import torch
import json
import pickle
import argparse
from sklearn.metrics import accuracy_score, f1_score

from src.model.gift_model import GIFTModel
from train import load_dataset, load_pickle, GIFTTrainer

def evaluate(model, Z_org, idx, true_labels):
    model.eval()
    with torch.no_grad():
        Z = torch.tensor(Z_org[idx], dtype=torch.float32)
        logits = model.classify(Z)           # just the classifier head, no loss
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    labels = [true_labels[i] for i in idx]
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

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
        "rank_ratio": 0.5,
        "temp": 0.5,
        "input_dim": 768,
        "num_classes": 2,
        "hidden_dim": 256,
        "projection_dim": 128,
        "eta": 0.5,
        "zeta": 0.5
    }

    # Rebuild Z_org (need graph + GCN + TD matrices to get text embeddings)
    trainer = GIFTTrainer(config)
    batch = {
        "texts": data["all_texts"],
        "tokens": tokens,
        "entities": entities,
        "pos": pos,
        "vocab": vocab,
        "labels_tensor": torch.tensor(data["all_labels"], dtype=torch.long),
        "labeled_idx": data["train_labeled_idx"],
    }

    # Load saved weights into model
    trainer.model.load_state_dict(
        torch.load(f"saved_models/{args.dataset}_gift.pt")
    )
    trainer.model.eval()
    print("Loaded saved model weights.")

    # Get Z_org (need to re-run forward up to embeddings, not the full loss)
    word_graph, entity_graph, pos_graph = trainer.build_graphs(
        tokens, entities, pos
    )
    
    H_w, H_e, H_p = trainer.run_gcn(word_graph, entity_graph, pos_graph)
    M_w, M_e, M_p = trainer.build_td_matrices(
        data["all_texts"], vocab, entities, pos
    )
    
    Z_org, _ = trainer.build_embeddings(
        M_w, M_e, M_p, M_w, M_e, M_p, # Augmented view not needed for eval
        H_w, H_e, H_p
    )

    # Evaluate on test set only
    acc, f1 = evaluate(
        trainer.model,
        Z_org,
        data["test_idx"],
        data["all_labels"]
    )

    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Test Macro-F1: {f1*100:.2f}%")