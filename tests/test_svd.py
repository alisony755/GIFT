import numpy as np
from src.representation.svd_views import SVDViewGenerator

# Run with python3 -m tests.test_svd

M = np.random.rand(50,50)
svd = SVDViewGenerator(rank_ratio=0.5)

M_r = svd.build_augmented_matrix(M)

for ratio in [1.0, 0.5, 0.2, 0.1]:
    svd = SVDViewGenerator(rank_ratio=ratio)
    M_r = svd.build_augmented_matrix(M)

    error = np.linalg.norm(M - M_r)

    print(ratio, error)

print("Original:", M.shape)
print("Reduced:", M_r.shape)

# print("Original matrix:")
# print(M)

# print("\nReconstructed matrix:")
# print(M_r)

# print("\nError:")
# print(np.linalg.norm(M - M_r))

print("Original norm:",
      np.linalg.norm(M))

print("Reconstruction error:",
      np.linalg.norm(M - M_r))

print("Relative error:",
      np.linalg.norm(M - M_r) /
      np.linalg.norm(M))