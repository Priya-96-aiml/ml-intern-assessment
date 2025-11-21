import unittest
import numpy as np
import sys, os

# Add src folder to path
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
        np.testing.assert_almost_equal(output[0,0], [5,0])
        np.testing.assert_almost_equal(output[0,1], [10,0])
        np.testing.assert_almost_equal(np.sum(weights, axis=-1), np.ones(weights.shape[:-1]))

    def test_attention_with_mask(self):
        mask = create_mask(2, 2, 'causal')
        output, weights = scaled_dot_product_attention(self.Q, self.K, self.V, mask)
        np.testing.assert_almost_equal(output.shape, (1,2,2))
        self.assertEqual(weights[0,0,1], 0.0)  # Masked position

    def test_batch_attention(self):
        Q_batch = np.array([self.Q[0], self.Q[0]])
        K_batch = np.array([self.K[0], self.K[0]])
        V_batch = np.array([self.V[0], self.V[0]])
        output, weights = scaled_dot_product_attention(Q_batch, K_batch, V_batch)
        self.assertEqual(output.shape, (2,2,2))
        self.assertEqual(weights.shape, (2,2,2))

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)

