import numpy as np

class SVD:
    def __init__(self, k=100, iters=20):
        self.k = k
        self.iters = iters

    def _power_iteration(self, A):
        b = np.random.randn(A.shape[1])
        b = b / (np.linalg.norm(b) + 1e-12)

        for _ in range(self.iters):
            b = A.T @ (A @ b)
            b = b / (np.linalg.norm(b) + 1e-12)

        sigma = np.linalg.norm(A @ b)
        u = (A @ b) / (sigma + 1e-12)

        return sigma, u, b

    def fit_transform(self, M):
        M = M.astype(np.float64)
        U_list, S_list, V_list = [], [], []

        A = M.copy()

        for _ in range(self.k):
            sigma, u, v = self._power_iteration(A)

            U_list.append(u)
            V_list.append(v)
            S_list.append(sigma)

            A = A - sigma * np.outer(u, v)

        U = np.stack(U_list, axis=1)
        V = np.stack(V_list, axis=1)
        S = np.diag(S_list)

        return U, S, V