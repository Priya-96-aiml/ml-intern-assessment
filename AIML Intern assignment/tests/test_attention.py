# AIML_Intern_assignment/tests/test_attention.py

import unittest
import numpy as np
import sys, os

# Add src folder to Python path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Import attention and utils
from attention import scaled_dot_product_attention
from utils import create_mask

class TestScaledDotProductAttention(unittest.TestCase):

    def setUp(self):
        # Batch size 1, sequence length 2, depth 2
        self.Q = np.array([[[1, 0], [0, 1]]], dtype=float)
        self.K = np.array([[[1, 0], [0, 1]]], dtype=float)
        self.V = np.array([[[5, 0], [10, 0]]], dtype=float)

    def test_basic_attention(self):
        """Test standard attention computation without mask"""
        output, weights = scaled_dot_product_attention(self.Q, self.K, self.V)
        
        # Check shape
        self.assertEqual(output.shape, self.V.shape)
        
        # Attention weights sum to 1 along last axis
        np.testing.assert_almost_equal(np.sum(weights, axis=-1), np.ones(weights.shape[:-1]))
        
        # Check output values are finite
        self.assertTrue(np.all(np.isfinite(output)))

    def test_attention_with_mask(self):
        """Test attention computation with causal mask"""
        mask = create_mask(2, 2, 'causal')
        output, weights = scaled_dot_product_attention(self.Q, self.K, self.V, mask)
        
        # Check shapes
        self.assertEqual(output.shape, (1, 2, 2))
        self.assertEqual(weights.shape, (1, 2, 2))
        
        # Masked positions should be zero
        self.assertEqual(weights[0, 0, 1], 0.0)

    def test_batch_attention(self):
        """Test attention computation with batch size > 1"""
        Q_batch = np.array([self.Q[0], self.Q[0]])
        K_batch = np.array([self.K[0], self.K[0]])
        V_batch = np.array([self.V[0], self.V[0]])
        
        output, weights = scaled_dot_product_attention(Q_batch, K_batch, V_batch)
        
        # Check shapes
        self.assertEqual(output.shape, (2, 2, 2))
        self.assertEqual(weights.shape, (2, 2, 2))
        
        # Check all outputs are finite
        self.assertTrue(np.all(np.isfinite(output)))

if __name__ == "__main__":
    # argv=[''] avoids Jupyter/Colab args interfering
    unittest.main(argv=[''], exit=False)

