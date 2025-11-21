# AIML Intern Assignment – Scaled Dot-Product Attention

This repository contains the **Scaled Dot-Product Attention** implementation and a **Trigram Language Model** as part of the AIML internship assignment. The project focuses on **clarity, modularity, and testability**, fully compatible with Google Colab.

---

## 📂 Project Structure

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


---

## ⚙️ How to Run

1. **Clone the repository**:
```bash
!git clone https://github.com/Priya-96-aiml/ml-intern-assessment.git
%cd "ml-intern-assignment/AIML Intern assignment"


Install dependencies:

!pip install numpy pytest


Run Attention in Colab:

from src.attention import scaled_dot_product_attention
from src.utils import create_mask
import numpy as np

# Example input
Q = np.array([[[1, 2], [3, 4]]])
K = np.array([[[1, 2], [3, 4]]])
V = np.array([[[1, 2], [3, 4]]])

# Compute attention
out, weights = scaled_dot_product_attention(Q, K, V)
print("Attention Output:\n", out)
print("Attention Weights:\n", weights)

# Optional: Use causal mask
mask = create_mask(Q.shape[1], mode="causal")
out_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=mask)
print("Attention Output with Causal Mask:\n", out_masked)
print("Attention Weights with Causal Mask:\n", weights_masked)


Run unit tests:

!pytest -q


✅ All tests should pass successfully.

📝 Design Highlights

Causal & Non-Causal Masks: Enables correct behavior for autoregressive tasks.

Numerical Stability: Scaled by sqrt(d_k) to prevent softmax saturation.

Trigram Language Model: Generates text probabilistically based on bigram context.

Robust Utilities: Includes text cleaning, tokenization, and reproducible seed setting.

Modular & Testable: Clean, maintainable, and unit-tested code.

🚀 Features & Learning Outcome

Implemented attention mechanism from scratch with hands-on Python & NumPy.

Designed reproducible, modular, and production-ready code.

Learned to handle edge cases, including empty inputs and unseen bigrams.

Built unit tests to validate core functionalities.

Fully compatible with Google Colab, ready for experimentation.

📄 Evaluation

For detailed explanations of design decisions and reasoning, see evaluation.md
.
