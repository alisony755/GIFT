import numpy as np

def cross_entropy(logits, labels):
    # Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Negative log likelihood
    n = logits.shape[0]
    loss = -np.sum(np.log(probs[np.arange(n), labels])) / n

    return loss