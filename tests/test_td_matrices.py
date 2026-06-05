from src.text_representation.td_matrix_builder import TDMatrixBuilder

# Run with python3 -m tests.test_td_matrices

texts = ["apple banana", "banana orange", "apple orange"]
vocab = ["apple", "banana", "orange"]

builder = TDMatrixBuilder()

M_w = builder.build_word_matrix(texts, vocab)

print("M_w shape:", M_w.shape)
print(M_w)