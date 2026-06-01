import numpy as np
import pickle

def load_glove_txt(path):
    embeddings = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vec

    return embeddings


if __name__ == "__main__":
    glove_path = "data/external/glove/glove.6B.100d.txt"
    out_path = "data/external/glove/glove.pkl"

    print("Loading GloVe...")
    emb = load_glove_txt(glove_path)

    print("Saving pickle...")
    with open(out_path, "wb") as f:
        pickle.dump(emb, f)

    print("Done:", out_path)