import numpy as np

class SVD:
    def __init__(self, k=15, n_iter=4, oversample=10, random_state=42):
        self.k = k
        self.n_iter = n_iter # Power iteration steps for accuracy on low-rank signal
        self.oversample = oversample # Adds extra dimensions for accuracy, then truncates
        self.random_state = random_state

    def fit_transform(self, M):
        rng = np.random.default_rng(self.random_state)
        M = M.astype(np.float64)
        n_rows, n_cols = M.shape
        k = min(self.k, min(n_rows, n_cols)) # Can't exceed matrix rank
        l = k + self.oversample # Oversampled rank

        # Random Gaussian projection matrix
        Omega = rng.standard_normal((n_cols, l))

        Y = M @ Omega

        # Power iteration to improve accuracy, so noisy singular values don't
        # leak into the approximation
        for _ in range(self.n_iter):
            Y = M @ (M.T @ Y)

        # QR decomposition
        Q, _ = np.linalg.qr(Y)

        # Project M into the subspace
        B = Q.T @ M

        # SVD on matrix B
        U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)

        # Recover left singular vectors in original space
        U = Q @ U_hat

        # Truncate to k and return in same format as before
        U = U[:, :k]
        S = np.diag(S[:k])
        V = Vt[:k, :].T

        return U, S, V