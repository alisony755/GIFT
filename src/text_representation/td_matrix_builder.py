import numpy as np

class TDMatrixBuilder:
    # Creates word TD matrix (M_w)
    def build_word_matrix(self, texts, vocab):
        M = np.zeros((len(texts), len(vocab)), dtype=np.float32)

        word2id = {w:i for i, w in enumerate(vocab)}

        for i, text in enumerate(texts):
            for w in text.split():
                if w in word2id:
                    M[i, word2id[w]] += 1

        return M

    # Creates entity binary matrix (M_e)
    def build_entity_matrix(self, texts, entities):
        M = np.zeros((len(texts), len(entities)), dtype=np.float32)
        ent2id = {e:i for i, e in enumerate(entities)}

        for i, text in enumerate(texts):
            for e in entities:
                if e in text:
                    M[i, ent2id[e]] = 1

        return M

    # Create POS TD matrix (M_p)
    def build_pos_matrix(self, pos_lists, pos_vocab):
        M = np.zeros((len(pos_lists), len(pos_vocab)), dtype=np.float32)

        pos2id = {p:i for i, p in enumerate(pos_vocab)}

        for i, tags in enumerate(pos_lists):
            for p in tags:
                if p in pos2id:
                    M[i, pos2id[p]] += 1

        return M