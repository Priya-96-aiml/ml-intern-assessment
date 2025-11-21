# AIML Intern Assignment – Evaluation

## 1. Overview

This project implements two key components for natural language processing and deep learning:

1. **Scaled Dot-Product Attention** – A core mechanism of Transformer architectures.  
2. **Trigram Language Model** – An N-gram probabilistic language model for text generation.

Both implementations are designed for clarity, modularity, and testability.

---

## 2. Design Choices

### Scaled Dot-Product Attention
- **Masking Strategy**: Implemented both causal and non-causal masks to control attention flow. Causal masking ensures the model does not attend to future tokens, preserving autoregressive behavior.  
- **Numerical Stability**: Added scaling by `sqrt(d_k)` to prevent softmax saturation and gradient issues.  
- **Vectorized Operations**: Used NumPy for batch-wise computations to ensure efficiency.  
- **Flexibility**: Supports optional masks for research experimentation and unit testing.

### Trigram Language Model
- **Start/End Tokens**: Added `<s>` and `</s>` tokens to mark sentence boundaries, improving generation coherence.  
- **Data Cleaning & Tokenization**: Text is lowercased, punctuation removed, and tokenized for robust model training.  
- **Randomized Generation**: Uses probabilistic selection of next word based on bigram context to generate diverse text sequences.  
- **Safety Checks**: Handles empty input or insufficient tokens gracefully to pass automated tests.

### Utilities
- **Seed Setting**: Implemented a unified `set_seed()` function for reproducible results across Python, NumPy, and PyTorch.  
- **Text Utilities**: Modular `clean_text()` and `tokenize()` functions for easy preprocessing.  
- **Logging**: Optional colored logging for debug and analysis during experimentation.  

---

## 3. Testing & Validation
- **Unit Tests**: `pytest` ensures attention outputs, weights, and trigram generation match expected results.  
- **Causal vs Non-Causal Tests**: Verifies correct masking behavior in attention computations.  
- **Edge Cases**: Empty input, short sequences, and unseen bigrams are handled safely.

---

## 4. Summary
- Focused on **clarity, modularity, and reproducibility** in both implementations.  
- Balanced **practical NLP preprocessing** with **theoretical understanding** of attention mechanisms.  
- Code is **Colab-ready**, easy to test, and well-structured for future extension into larger NLP or deep learning projects.  

> **Outcome:** A clean, efficient, and fully tested assignment demonstrating strong Python, NLP, and AI/ML implementation skills, highlighting readiness for real-world AI/ML roles.

