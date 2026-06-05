class VocabularyBuilder:
    def build(self, tokenized_texts):
        vocab = sorted(
            list(
                set(
                    token
                    for doc in tokenized_texts
                    for token in doc
                )
            )
        )

        return vocab
