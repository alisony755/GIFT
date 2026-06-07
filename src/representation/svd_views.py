import numpy as np

# Truncated SVD view generation
class SVDViewGenerator:
    def __init__(self, k=15, n_iter=4, oversample=10, random_state=42):
        self.k = k
        self.n_iter = n_iter
        self.oversample = oversample
        self.random_state = random_state

    def svd(self, M, k):
        rng = np.random.default_rng(self.random_state)

        if hasattr(M, 'toarray'):
            M = M.toarray()

        M = M.astype(np.float64)
        n_rows, n_cols = M.shape
        k = min(k, min(n_rows, n_cols))
        l = k + self.oversample

        # Random Gaussian projection
        Omega = rng.standard_normal((n_cols, l))
        Y = M @ Omega

        # Power iteration
        for _ in range(self.n_iter):
            Y = M @ (M.T @ Y)

        # QR decomposition
        Q, _ = np.linalg.qr(Y)

        # Project into small subspace
        B = Q.T @ M

        # Cheap SVD on small matrix
        U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)

        # Recover left singular vectors
        U = Q @ U_hat

        U = U[:, :k]
        S_diag = np.diag(S[:k])
        V = Vt[:k, :].T

        return U, S_diag, Vt[:k, :]

    def reconstruct(self, U, S, Vt):
        return U @ S @ Vt

    def build_augmented_matrix(self, M):
        U, S, Vt = self.svd(M, self.k)
        return self.reconstruct(U, S, Vt)