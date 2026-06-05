import numpy as np
from src.text_representation.text_encoder import TextEncoder

# Run with python3 -m tests.test_text_representation

encoder = TextEncoder()

M_w = np.array([[1,0],[0,1]])
M_e = np.array([[1,1],[0,1]])
M_p = np.array([[1,0],[1,1]])

H_w = np.random.randn(2, 4)
H_e = np.random.randn(2, 4)
H_p = np.random.randn(2, 4)

Z_org = encoder.build_original(M_w, M_e, M_p, H_w, H_e, H_p)
Z_aug = encoder.build_augmented(M_w, M_e, M_p, H_w, H_e, H_p)
Z_final = encoder.build_final(Z_org, Z_aug)

print("Z_org:", Z_org.shape)
print("Z_aug:", Z_aug.shape)
print("Z_final:", Z_final.shape)