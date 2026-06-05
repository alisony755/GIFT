import re

class EntityExtractor:
    def __init__(self, ent2ids):
        # ent2ids
        self.ent2ids = ent2ids
        self.entities = list(ent2ids.keys())

    # Extract entities per document
    def extract(self, texts):
        results = []

        for i, text in enumerate(texts):
            if i % 50 == 0:
                print(f"[EntityExtractor] processing doc {i}/{len(texts)}")

            lower = text.lower()
            doc_entities = []

            # Strict match only
            for entity in self.ent2ids.keys():
                # Must match whole entity phrase
                if f" {entity} " in f" {lower} ":
                    doc_entities.append(entity)

            results.append(doc_entities)

        assert len(results) == len(texts)

        return results