import numpy as np
from embeddings.svd import SVD

class SVDAugmenter:
    def __init__(self, k=100):
        self.svd = SVD(k=k)

    def augment_matrix(self, M):
        U, S, V = self.svd.fit_transform(M)

        #runcated reconstruction
        return U @ S @ V.T

    def build_augmented_Z(self, M_w, M_e, M_p, H_w, H_e, H_p):
        M_w_r = self.augment_matrix(M_w)
        M_e_r = self.augment_matrix(M_e)
        M_p_r = self.augment_matrix(M_p)

        Z_w_r = M_w_r @ H_w
        Z_e_r = M_e_r @ H_e
        Z_p_r = M_p_r @ H_p

        Z_aug = np.concatenate([Z_w_r, Z_e_r, Z_p_r], axis=1)

        return Z_aug