import unittest
import numpy as np
import sys, os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from attention import scaled_dot_product_attention
from utils import create_mask

class TestScaledDotProductAttention(unittest.TestCase):

    def setUp(self):
        self.Q = np.array([[[1, 0], [0, 1]]], dtype=float)
        self.K = np.array([[[1, 0], [0, 1]]], dtype=float)
        self.V = np.array([[[5, 0], [10, 0]]], dtype=float)

    def test_basic_attention(self):
        output, weights = scaled_dot_product_attention(self.Q, self.K, self.V)
        np.testing.assert_almost_equal(output[0, 0], [5, 0])
        np.testing.assert_almost_equal(output[0, 1], [10, 0])
        np.testing.assert_almost_equal(np.sum(weights, axis=-1), np.ones(weights.shape[:-1]))

    def test_attention_with_mask(self):
        mask = create_mask(seq_q_len=2, seq_k_len=2, mask_type='causal')
        output, weights = scaled_dot_product_attention(self.Q, self.K, self.V, mask=mask)
        np.testing.assert_almost_equal(output.shape, (1,2,2))
        self.assertTrue(np.all(weights[:,0,1] == 0))  # Masked position = 0

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
