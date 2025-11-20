import numpy as np
from src.attention import scaled_dot_product_attention

def test_attention_shapes():
    """
    Unit test for shape correctness.
    """
    batch_size, seq_len, d_k, d_v = 2, 4, 8, 5
    Q = np.random.rand(batch_size, seq_len, d_k)
    K = np.random.rand(batch_size, seq_len, d_k)
    V = np.random.rand(batch_size, seq_len, d_v)
    
    output, attn = scaled_dot_product_attention(Q, K, V, return_weights=True)
    
    assert output.shape == (batch_size, seq_len, d_v), f"Output shape mismatch: {output.shape}"
    assert attn.shape == (batch_size, seq_len, seq_len), f"Attention shape mismatch: {attn.shape}"
    print("Test passed!")

def test_attention_masking():
    """
    Unit test for masking effect (output should ignore masked positions)
    """
    Q = np.random.rand(1, 3, 4)
    K = np.random.rand(1, 3, 4)
    V = np.random.rand(1, 3, 2)
    
    mask = np.array([[[1,1,0],
                      [1,1,0],
                      [1,1,0]]])
    
    output, attn = scaled_dot_product_attention(Q, K, V, mask=mask, return_weights=True)
    # Ensure masked weights are effectively zero
    assert np.all(attn[:,:,2] < 1e-5), "Masking not applied correctly"
    print("Masking test passed!")

if __name__ == "__main__":
    test_attention_shapes()
    test_attention_masking()

