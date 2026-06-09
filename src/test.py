import torch
import json
import argparse
from sklearn.metrics import accuracy_score, f1_score

from src.train import load_dataset, load_pickle, GIFTTrainer

def evaluate(model, Z_org, idx, true_labels):
    model.eval()
    with torch.no_grad():
        Z = Z_org[idx]
        logits = model.classify(Z)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = [true_labels[i] for i in idx]
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1

def rebuild_embeddings(trainer, word_graph, entity_graph, pos_graph, all_texts, vocab, entities, pos):
    # Convert graph inputs to tensors
    X_w = torch.tensor(word_graph.X, dtype=torch.float32)
    A_w = torch.tensor(word_graph.A, dtype=torch.float32)
    X_e = torch.tensor(entity_graph.X, dtype=torch.float32)
    A_e = torch.tensor(entity_graph.A, dtype=torch.float32)
    X_p = torch.tensor(pos_graph.X, dtype=torch.float32)
    A_p = torch.tensor(pos_graph.A, dtype=torch.float32)

    # Build TD matrices and convert to tensors
    M_w, M_e, M_p = trainer.build_td_matrices(all_texts, vocab, entities, pos)
    M_w_t = torch.tensor(M_w, dtype=torch.float32)
    M_e_t = torch.tensor(M_e, dtype=torch.float32)
    M_p_t = torch.tensor(M_p, dtype=torch.float32)

    with torch.no_grad():
        H_w, H_e, H_p = trainer.run_gcn(X_w, A_w, X_e, A_e, X_p, A_p)
        Z_org, _ = trainer.build_embeddings(
            M_w_t, M_e_t, M_p_t,
            M_w_t, M_e_t, M_p_t,
            H_w, H_e, H_p
        )
    return Z_org

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

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
        "input_dim": 384,
        "num_classes": 2,
        "hidden_dim": 128,
        "projection_dim": 128,
        "eta": 0.5,
        "zeta": 0.5,
        "batch_size": 256,
    }

    trainer = GIFTTrainer(config)
    word_graph, entity_graph, pos_graph = trainer.build_graphs(tokens, entities, pos)
    trainer.init_gcn(word_graph, entity_graph, pos_graph)

    # Load saved model weights
    checkpoint = torch.load(f"saved_models/{args.dataset}_gift.pt", weights_only=True)
    trainer.model.load_state_dict(checkpoint["model"], strict=True)
    trainer.gcn_w.load_state_dict(checkpoint["gcn_w"])
    trainer.gcn_e.load_state_dict(checkpoint["gcn_e"])
    trainer.gcn_p.load_state_dict(checkpoint["gcn_p"])
    print("Loaded saved model weights.")
    
    trainer.model.eval()
    trainer.gcn_w.eval()
    trainer.gcn_e.eval()
    trainer.gcn_p.eval()

    # Rebuild embeddings
    Z_org = rebuild_embeddings(
        trainer, word_graph, entity_graph, pos_graph,
        data["all_texts"], vocab, entities, pos
    )

    # Evaluate on test set
    acc, f1 = evaluate(
        trainer.model,
        Z_org,
        data["test_idx"],
        data["all_labels"]
    )

    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Test Macro-F1: {f1*100:.2f}%")