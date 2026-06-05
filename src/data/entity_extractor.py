class EntityExtractor:
    def __init__(self, entity_vocab):
        self.entity_vocab = entity_vocab

    def extract(self, texts):
        results = []

        for text in texts:
            found = []

            lower = text.lower()

            for entity in self.entity_vocab:
                if entity.lower() in lower:
                    found.append(entity)

            results.append(found)

        return results