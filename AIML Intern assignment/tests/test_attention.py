import unittest
import numpy as np
import sys
import os

# Add src folder to Python path for Colab
sys.path.append(os.path.join(os.getcwd(), "src"))

from attention import scaled_dot_product_attention

class TestScaledDotProductAttention(unittest.TestCase):

    def setUp(self):
        # Simple batch of 1, sequence length 2, depth 2
        self.Q = np.array([[[1, 0], [0, 1]]], dtype=float)
        self.K = np.array([[[1, 0], [0, 1]]], dtype=float)
        self.V = np.array([[[5, 0], [10, 0]]], dtype=float)

    def test_basic_attention(self):
        output, weights = scaled_dot_product_attention(self.Q, self.K, self.V)
        # Each query attends mostly to matching key
        np.testing.assert_almost_equal(output[0, 0], [5, 0])
        np.testing.assert_almost_equal(output[0, 1], [10, 0])
        # Sum of weights along keys = 1 for each query
        np.testing.assert_almost_equal(np.sum(weights, axis=-1), np.ones(weights.shape[:-1]))

    def test_attention_with_mask(self):
        mask = np.array([[[0, 1], [0, 0]]])  # Mask second key for first query
        output, weights = scaled_dot_product_attention(self.Q, self.K, self.V, mask=mask)
        np.testing.assert_almost_equal(output[0, 0], [5, 0])  # Only first value attended
        np.testing.assert_almost_equal(weights[0, 0], [1.0, 0.0])  # Masked position = 0

    def test_batch_attention(self):
        # Batch of 2
        Q = np.array([self.Q[0], self.Q[0]])
        K = np.array([self.K[0], self.K[0]])
        V = np.array([self.V[0], self.V[0]])
        output, weights = scaled_dot_product_attention(Q, K, V)
        self.assertEqual(output.shape, (2, 2, 2))
        self.assertEqual(weights.shape, (2, 2, 2))

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)  # exit=False to run in Colab

