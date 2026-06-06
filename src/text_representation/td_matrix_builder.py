import numpy as np

class TDMatrixBuilder:
    # Creates word TD matrix (M_w)
    def build_word_matrix(self, texts, vocab):
        vocab_list = sorted(list(vocab))
        word2id = {
            w: i
            for i, w in enumerate(vocab_list)
        }

        M = np.zeros(
            (len(texts), len(vocab_list)),
            dtype=np.float32
        )

        for doc_idx, text in enumerate(texts):
            for word in text.lower().split():
                if word in word2id:
                    M[doc_idx][word2id[word]] = 1.0

        return M

    # Creates entity binary matrix (M_e)
    def build_entity_matrix(self, texts, entities):
        n_docs = len(texts)
        entities = entities[:n_docs]

        all_entities = sorted(list(set(
            e for doc in entities for e in doc
        )))

        ent2id = {e: i for i, e in enumerate(all_entities)}

        M = np.zeros((n_docs, len(all_entities)), dtype=np.float32)
        
        # DEBUG
        print("Texts:", len(texts))
        print("Entities:", len(entities))

        for doc_idx, doc_entities in enumerate(entities):
            for e in doc_entities:
                if e in ent2id:
                    M[doc_idx][ent2id[e]] = 1.0

        return M

    # Create POS TD matrix (M_p)
    def build_pos_matrix(self, pos_tags, pos_vocab):
        pos2id = {
            p: i
            for i, p in enumerate(pos_vocab)
        }

        # Force alignment check
        num_docs = len(pos_tags)

        M = np.zeros(
            (num_docs, len(pos_vocab)),
            dtype=np.float32
        )

        for doc_idx, tags in enumerate(pos_tags):
            # Prevent mismatch crashes
            if doc_idx >= num_docs:
                break

            # Ensure tags is iterable
            if isinstance(tags, str):
                tags = tags.split()

            for tag in tags:
                if tag in pos2id:
                    M[doc_idx][pos2id[tag]] = 1.0

        return M