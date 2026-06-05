import pickle
import re

input_path = "data/external/NELL_KG/ent2ids_refined"
output_path = "data/external/NELL_KG/entity_map.pkl"

entity_map = {}

pattern = re.compile(r'"(.*?)"\s*:\s*(\d+)')

with open(input_path, "r") as f:
    for line in f:
        matches = pattern.findall(line)

        for entity, idx in matches:
            entity_map[entity.lower()] = int(idx)

with open(output_path, "wb") as f:
    pickle.dump(entity_map, f)

print("Saved entities:", len(entity_map))