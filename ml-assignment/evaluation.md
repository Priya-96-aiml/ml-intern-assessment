# Evaluation – Trigram Language Model

## 1. Storage of N-Gram Counts

**Implemented as a Python dictionary:**

- **Key:** Tuple of the previous two words `(word1, word2)`  
- **Value:** List of potential next words `[word3, word3, ...]`  

**Rationale:**

- Enables efficient lookups  
- Supports probabilistic selection of next words for natural text generation  

---

## 2. Text Preprocessing & Handling Unknown Words

**Cleaning:**

- Converted to lowercase  
- Removed punctuation, URLs, emojis, and special characters  
- Optional stopword removal and lemmatization (NLTK)  

**Padding:**

- Added `<s>` tokens at start and `</s>` at end for sentence boundaries  

**Unknown Words:**

- Skipped during generation to maintain output consistency  

---

## 3. Text Generation & Probabilistic Sampling

**Starting Point:** Random or user-specified bigram  

**Next Word Selection:**

- Gather all words following current bigram  
- Randomly select one (probabilistic sampling)  

**Termination:**

- Upon generating `</s>` token or reaching maximum length  

**Outcome:**  

- Produces coherent and contextually relevant sentences  

---

## 4. Additional Design Decisions

**Edge Cases:**

- Empty input or very short text returns empty string  
- Stops gracefully if no matching trigram is found  

**Large Dataset Support:**

- Tested with corpora >100K characters  
- Optimized for memory efficiency and speed  

**Reproducibility:**  

- Optional random seed for deterministic outputs  

**Utilities (`utils.py`):**  

- Centralized cleaning, tokenization, and seed setup  

**Testing:**  

- Unit tests cover normal, empty, and short corpus scenarios  

---

## Summary

This project demonstrates:

- Hands-on NLP implementation skills  
- Ability to handle small and large datasets  
- Robust preprocessing and edge case handling  
- Probabilistic text generation for realistic outputs  
- Clean, maintainable, and Colab-ready code  

> The Trigram Language Model balances technical rigor with usability, ensuring high-quality outputs while being scalable and robust for various text corpora.
