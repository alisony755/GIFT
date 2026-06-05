import pickle

def load_ent2ids(path):
    with open(path, "rb") as f:
        return pickle.load(f)