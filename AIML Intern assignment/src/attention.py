import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None, causal=False, return_weights=False):
    """
    Compute Scaled Dot-Product Attention.

    Args:
        Q: Queries, shape (batch_size, seq_len, d_k)
        K: Keys, shape (batch_size, seq_len, d_k)
        V: Values, shape (batch_size, seq_len, d_v)
        mask: Optional mask (batch_size, seq_len, seq_len), 1 for valid positions, 0 for masked
        causal: Bool, if True applies causal mask to prevent attending to future positions
        return_weights: Bool, if True returns attention weights

    Returns:
        output: (batch_size, seq_len, d_v)
        attention_weights (optional): (batch_size, seq_len, seq_len)
    """
    batch_size, seq_len, d_k = Q.shape
    
    # Step 1: Compute raw attention scores
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)  # shape: (batch, seq_len, seq_len)
    
    # Step 2: Apply mask if provided
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    # Step 3: Apply causal mask to prevent future attention
    if causal:
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        scores = scores * causal_mask - 1e9 * (1 - causal_mask)
    
    # Step 4: Numerically stable softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    
    # Step 5: Weighted sum to get output
    output = np.matmul(attention_weights, V)
    
    if return_weights:
        return output, attention_weights
    return output

