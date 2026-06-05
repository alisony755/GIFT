import os
import pickle
import re

ent2ids_path = "data/external/NELL_KG/ent2ids_refined"
out_path = "data/processed/ent2ids_refined.pkl"

os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(ent2ids_path, "r", encoding="utf-8") as f:
    raw = f.read().strip()

# Extract all "entity": id pairs
pattern = r'"([^"]+)"\s*:\s*(\d+)'
matches = re.findall(pattern, raw)
ent2ids = {e.lower(): int(i) for e, i in matches}

print("Entities loaded:", len(ent2ids))

with open(out_path, "wb") as f:
    pickle.dump(ent2ids, f)