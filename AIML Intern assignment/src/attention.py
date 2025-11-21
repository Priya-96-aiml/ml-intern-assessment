import numpy as np
from utils import create_mask  # Import from same src folder

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute Scaled Dot-Product Attention.

    Args:
        Q: Queries (batch_size, seq_len_q, depth)
        K: Keys (batch_size, seq_len_k, depth)
        V: Values (batch_size, seq_len_v, depth_v)
        mask: Optional mask (batch_size, seq_len_q, seq_len_k)

    Returns:
        output: attended output (batch_size, seq_len_q, depth_v)
        attention_weights: softmax attention weights (batch_size, seq_len_q, seq_len_k)
    """
    scores = np.matmul(Q, K.transpose(0, 2, 1))
    dk = K.shape[-1]
    scores /= np.sqrt(dk)

    if mask is not None:
        scores = np.where(mask == 1, -1e9, scores)

    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    output = np.matmul(attention_weights, V)
    return output, attention_weights


# Demo
if __name__ == "__main__":
    Q = np.random.rand(1, 3, 4)
    K = np.random.rand(1, 3, 4)
    V = np.random.rand(1, 3, 4)
    mask = create_mask(3, 3)

    output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
    print("Attention Output:\n", output)
    print("Attention Weights:\n", attn_weights)
