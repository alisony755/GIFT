import numpy as np

class TextRepresentationBuilder:
    def __init__(self):
        pass

    def aggregate(self, M, H):
        return M @ H

    def build_original_views(self, M_w, M_e, M_p, H_w, H_e, H_p):
        Z_w = self.aggregate(M_w, H_w)
        Z_e = self.aggregate(M_e, H_e)
        Z_p = self.aggregate(M_p, H_p)

        Z_org = np.concatenate([Z_w, Z_e, Z_p], axis=1)

        return Z_org

    def build_augmented_views(self, M_wr, M_er, M_pr, H_w, H_e, H_p):
        Z_wr = self.aggregate(M_wr, H_w)
        Z_er = self.aggregate(M_er, H_e)
        Z_pr = self.aggregate(M_pr, H_p)

        Z_aug = np.concatenate([Z_wr, Z_er, Z_pr], axis=1)

        return Z_aug

    def build_final(self, Z_org, Z_aug):
        return np.concatenate([Z_org, Z_aug], axis=1)