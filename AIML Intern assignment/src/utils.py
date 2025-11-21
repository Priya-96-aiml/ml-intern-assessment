import numpy as np

def create_mask(seq_q_len, seq_k_len, mask_type='causal'):
    """
    Create a mask for attention.

    Args:
        seq_q_len: Length of queries
        seq_k_len: Length of keys
        mask_type: 'causal' or None
    Returns:
        mask: numpy array of shape (1, seq_q_len, seq_k_len)
    """
    if mask_type == 'causal':
        mask = np.triu(np.ones((1, seq_q_len, seq_k_len)), k=1)
    else:
        mask = np.zeros((1, seq_q_len, seq_k_len))
    return mask.astype(np.int32)
