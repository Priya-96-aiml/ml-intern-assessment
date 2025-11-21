# AIML Intern Assignment – Scaled Dot-Product Attention

This repository contains the **Scaled Dot-Product Attention** implementation and a **Trigram Language Model** as part of the AIML internship assignment. The project focuses on **clarity, modularity, and testability**, fully compatible with Google Colab.

## Scaled Dot-Product Attention

### Core Idea
- Attention allows a model to focus on relevant parts of the input sequence when generating outputs.
- Scaled Dot-Product Attention computes weights between queries (Q) and keys (K) and applies them to values (V) to get a context-aware output.

### Mathematical Formula

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
\]

Where:  
- **Q** = Query matrix  
- **K** = Key matrix  
- **V** = Value matrix  
- **dₖ** = Dimension of keys (used to scale the dot product for numerical stability)

### Why Scale by √dₖ
- Prevents large dot-product values that can make softmax produce extremely small gradients.  
- Improves numerical stability during training.

### Masking
- **Causal Mask**: Prevents attending to future tokens in autoregressive tasks (like language generation).  
- **Padding Mask**: Ignores padding tokens in variable-length sequences.

### Steps in Computation
1. Compute dot product between Q and Kᵀ.  
2. Scale by √dₖ.  
3. Apply softmax to get attention weights.  
4. Multiply weights with V to get the final attention output.

### Benefits
- Captures long-range dependencies in sequences.  
- Fully differentiable – allows end-to-end training.  
- Efficient computation with matrix operations.

##  Project Structure

    AIML_Intern_assignment/
      ├── src/ # Source code
      │ ├── attention.py # Scaled Dot-Product Attention
      │ ├── utils.py # Utilities: masking, tokenization, seed setting
      │ └── init.py
      ├── tests/ # Unit tests
      │ └── test_attention.py
      ├── notebooks/ # Optional experimentation notebooks
      ├── README.md
      └── evaluation.md # Design choices and project evaluation


-

##  How to Run

1. **Clone the repository**:


         !git clone https://github.com/Priya-96-aiml/ml-intern-assessment.git
         %cd "ml-intern-assignment/AIML Intern assignment"


Install dependencies:

    !pip install numpy pytest


Run Attention in Colab:

    from src.attention import scaled_dot_product_attention
    from src.utils import create_mask
    import numpy as np

### Example input
    Q = np.array([[[1, 2], [3, 4]]])
    K = np.array([[[1, 2], [3, 4]]])
    V = np.array([[[1, 2], [3, 4]]])

### Compute attention
    out, weights = scaled_dot_product_attention(Q, K, V)
    print("Attention Output:\n", out)
    print("Attention Weights:\n", weights)

### Optional: Use causal mask
    mask = create_mask(Q.shape[1], mode="causal")
    out_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=mask)
    print("Attention Output with Causal Mask:\n", out_masked)
    print("Attention Weights with Causal Mask:\n", weights_masked)


Run unit tests:

    !pytest -q


### All tests should pass successfully.

## Design Highlights

- Causal & Non-Causal Masks: Enables correct behavior for autoregressive tasks.

- Numerical Stability: Scaled by sqrt(d_k) to prevent softmax saturation.

- Trigram Language Model: Generates text probabilistically based on bigram context.

- Robust Utilities: Includes text cleaning, tokenization, and reproducible seed setting.

- Modular & Testable: Clean, maintainable, and unit-tested code.

## Features & Learning Outcome

- Implemented attention mechanism from scratch with hands-on Python & NumPy.

- Designed reproducible, modular, and production-ready code.

- Learned to handle edge cases, including empty inputs and unseen bigrams.

- Built unit tests to validate core functionalities.

- Fully compatible with Google Colab, ready for experimentation.

## Evaluation

- For detailed explanations of design decisions and reasoning, see evaluation.md.
