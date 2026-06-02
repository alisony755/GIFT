import numpy as np
import ast

# Loads NELL TransE embeddings for entity graph node features
class TransELoader:
    def __init__(self, embedding_file, mapping_file):
        # Embeddings (row-aligned matrix)
        self.embeddings = np.loadtxt(embedding_file, dtype=np.float32)

        self.embedding_dim = self.embeddings.shape[1]

        # Handle single-line dict format
        with open(mapping_file, "r") as f:
            raw = f.read().strip()

        # Remove whitespaces and trailing commas
        raw = raw.strip().rstrip(",")

        # Convert string dict to Python dict
        try:
            self.entity_to_id = {}
            with open(mapping_file, "r") as f:
                raw = f.read().strip()

            # Remove outer braces if present
            raw = raw.strip()
            if raw.startswith("{") and raw.endswith("}"):
                raw = raw[1:-1]

            # Split entries safely
            for item in raw.split(","):
                item = item.strip()

                if not item:
                    continue

                # Assumes each item looks like: "name": 123
                if ":" not in item:
                    continue

                key, val = item.split(":", 1)

                key = key.strip().strip('"').strip("'")
                val = val.strip()

                # Remove trailing }
                val = val.replace("}", "").strip()

                try:
                    self.entity_to_id[key] = int(val)
                except ValueError:
                    continue
                
        except Exception:
            # Fallback manual parse
            self.entity_to_id = {}
            items = raw.split(",")
            for item in items:
                if ":" in item:
                    k, v = item.split(":")
                    k = k.strip().strip('"')
                    v = int(v.strip())
                    self.entity_to_id[k] = v

    def get_embedding(self, entity):
        idx = self.entity_to_id.get(entity)

        if idx is None or idx >= self.embeddings.shape[0]:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        return self.embeddings[idx]

    def build_feature_matrix(self, entities):
        return np.array(
            [self.get_embedding(e) for e in entities],
            dtype=np.float32
        )