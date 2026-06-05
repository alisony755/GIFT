import re

class EntityExtractor:
    def __init__(self, ent2ids):
        # ent2ids
        self.ent2ids = ent2ids
        self.entities = list(ent2ids.keys())

    def extract(self, texts):
        results = []

        for text in texts:
            lower = text.lower()

            found = []

            for entity in self.entities:
                # Substring match
                if re.search(r"\b" + re.escape(entity.lower()) + r"\b", lower):
                    found.append(entity)

            results.append(found)

        return results