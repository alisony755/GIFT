import numpy as np

# Truncated SVD view generation
class SVDViewGenerator:
    def __init__(self, rank_ratio=0.1):
        self.rank_ratio = rank_ratio

    def svd(self, M):
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        return U, S, Vt

    def truncate(self, U, S, Vt, r):
        U_r = U[:, :r]
        S_r = np.diag(S[:r])
        Vt_r = Vt[:r, :]
        return U_r, S_r, Vt_r

    def reconstruct(self, U_r, S_r, Vt_r):
        return U_r @ S_r @ Vt_r

    def build_augmented_matrix(self, M):
        U, S, Vt = self.svd(M)

        r = max(1, int(S.shape[0] * self.rank_ratio))
        U_r, S_r, Vt_r = self.truncate(U, S, Vt, r)

        M_r = self.reconstruct(U_r, S_r, Vt_r)

        return M_r