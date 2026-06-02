import numpy as np

# Truncated SVD view generation
class SVDViewGenerator:
    def __init__(
        self,
        rank_ratio=0.1,
        max_iter=100,
        tol=1e-6
    ):
        self.rank_ratio = rank_ratio
        self.max_iter = max_iter
        self.tol = tol

    # Compute dominant eigenpair using power iteration
    def power_iteration(self, A):
        n = A.shape[0]

        v = np.random.rand(n)
        v = v / np.linalg.norm(v)

        for _ in range(self.max_iter):
            v_new = A @ v
            v_new = v_new / np.linalg.norm(v_new)

            if np.linalg.norm(v_new - v) < self.tol:
                break

            v = v_new

        eigenvalue = v.T @ A @ v

        return eigenvalue, v

    # Compute top-r eigenpairs using deflation
    def eigendecompose(self, A, r):
        eigenvalues = []
        eigenvectors = []

        A_work = A.copy()

        for _ in range(r):
            value, vector = self.power_iteration(A_work)

            eigenvalues.append(value)
            eigenvectors.append(vector)

            # Deflation
            A_work = (A_work - value * np.outer(vector, vector))

        return (
            np.array(eigenvalues),
            np.column_stack(eigenvectors)
        )

    # Compute SVD from eigen-decomposition
    def svd(self, M, r):
        # Compute covariance matrix
        MtM = M.T @ M

        # Compute top-r eigenpairs
        eigenvalues, V = self.eigendecompose(MtM, r)

        # Compute singular values
        singular_values = np.sqrt(np.maximum(eigenvalues, 0))

        # Construct diagonal matrix
        S = np.diag(singular_values)

        # Compute left singular vectors
        U = np.zeros((M.shape[0], r))

        for i in range(r):
            sigma = singular_values[i]

            if sigma > 1e-12:
                U[:, i] = (M @ V[:, i]) / sigma

        Vt = V.T

        return U, S, Vt

    # Reconstruct low-rank approximation
    def reconstruct(self, U, S, Vt):
        return U @ S @ Vt

    # Build SVD-augmented matrix
    def build_augmented_matrix(self, M):
        max_rank = min(M.shape)

        r = max(1, int(max_rank * self.rank_ratio))

        U, S, Vt = self.svd(M, r)
        M_r = self.reconstruct(U, S, Vt)

        return M_r