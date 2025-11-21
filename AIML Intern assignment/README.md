# Trigram Language Model — ML Intern Assessment

This repository contains the **Trigram Language Model** assignment, part of the ML Intern Assessment.  
It demonstrates the implementation of an n-gram language model capable of generating meaningful text sequences while highlighting **technical proficiency, clean coding practices, and problem-solving skills**.

---

## 📂 Project Structure

   ml-assignment/
   │
   ├─ data/ → Dataset files used for training/testing
   ├─ src/ → Source code
   │ └─ generate.py → Main trigram model implementation
   ├─ tests/ → Unit tests ensuring correctness
   ├─ README.md → This file
   └─ evaluation.md → Design choices and technical decisions summary



## 🔹 Project Overview

The **Trigram Language Model** project focuses on:

- Text preprocessing & tokenization  
- Building trigram counts from the dataset  
- Generating text sequences using trigram probabilities  
- Handling unknown words and sentence boundaries for robustness  

**Key Highlights:**

- Efficient **nested dictionary** data structure for storing trigrams  
- **Weighted random selection** for realistic text generation  
- Modular, testable Python code following **software engineering best practices**  
- Unit tests to ensure correctness and maintainability  

---

## 🔧 How to Run

1. **Clone the repository**

git clone https://github.com/VarshiniAG/ml-intern-assessment.git
cd ml-assignment
Create a virtual environment (optional)

Copy code
python3 -m venv venv
# Activate the environment
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows
Install dependencies


pip install -r requirements.txt
Run the trigram model


python src/generate.py
Run tests


python -m unittest discover tests

## Design Highlights
Tokenization: Lowercasing, punctuation removal, whitespace splitting

Unknown Words: Handled with <UNK> token for robustness

Sentence Boundaries: <START> and <END> tokens

Data Structure: Nested dictionaries for fast trigram lookup

Text Generation: Conditional probabilities with weighted random selection

Modularity: Functions for preprocessing, training, and generation

Testing: Unit tests ensure correctness and maintainability

Full reasoning and technical decisions are documented in evaluation.md.

## Key Takeaways & Skills Demonstrated
Strong grasp of language modeling and n-gram probabilities

Proficient in Python coding, modular design, and testing

Emphasis on robustness and maintainability

Prepared for real-world NLP and AI/ML projects

 ## Future Enhancements
Higher-order n-grams and smoothing techniques for improved predictions

Train on larger, more diverse datasets for richer outputs

Deploy as a web application for interactive text generation

Optimize for performance and scalability on large datasets

## Conclusion
This project demonstrates a working, efficient, and robust trigram language model, highlighting strategic thinking, attention to detail, and professional readiness.
It is designed to impress recruiters and demonstrate both technical competence and business-ready problem-solving skills.


