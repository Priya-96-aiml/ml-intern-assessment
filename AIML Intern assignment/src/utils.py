import numpy as np

def create_padding_mask(seq, pad_value=0):
    """
    Create a mask for padding tokens.

    Args:
        seq (np.array): Input sequence, shape (batch_size, seq_len)
        pad_value (int): Value representing padding (default 0)

    Returns:
        mask (np.array): Mask of shape (batch_size, 1, seq_len)
    """
    return (seq != pad_value).astype(np.float32)[:, np.newaxis, :]

