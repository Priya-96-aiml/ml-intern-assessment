## Trigram Language Model

This repository implements a Trigram Language Model in Python. The model predicts the next word based on the previous two words and can generate meaningful text sequences. It works efficiently on both small and large datasets.

## How to Run
# 1. Clone Repository
       # git clone https://github.com/Priya-96-aiml/ml-intern-assessment.git
       # cd ml-intern-assessment/ml-assignment

# 2. Run on Small Corpus
         # python3 src/generate.py
 * Uses data/example_corpus.txt
 * Trains the trigram model
 * Generates 30-word sample text

# 3. Run on Large Dataset (Colab Recommended)

Upload or Mount Dataset

       # from google.colab import drive
        drive.mount('/content/drive')
        file_path = "/content/drive/MyDrive/large_dataset.txt"


Load, Train, and Generate Text

from src.ngram_model import TrigramModel

with open(file_path, "r", encoding="utf-8") as f:
    corpus = f.read()

model = TrigramModel()
model.fit(corpus)

generated_text = model.generate(max_words=100)
print("\nGenerated Text:\n", generated_text)


Works with datasets of any size

Output length customizable with max_words

Design Choices & Highlights

Trigram Approach

Uses the previous two words to predict the next

Provides a balance between context awareness and efficiency

Robust Text Preprocessing

Lowercasing, punctuation removal

Optional stopword removal & lemmatization (via NLTK)

Handles emojis, URLs, and special characters

Randomized Generation

Selects the next word probabilistically for natural variation

Ensures generated text is non-repetitive and diverse

Edge Case Handling

Returns empty string for empty or very short text

Ensures stability on all input types

Large Dataset Support

Optimized for large corpora (e.g., books from Project Gutenberg)

Tested with “Alice in Wonderland” (~150K characters)

Testing & Reliability

Unit tests for standard, empty, and short texts

Guarantees correct behavior before deployment

Reusable Utilities

utils.py provides reusable cleaning and tokenization functions

Optional seed utilities ensure reproducible results

## Folder Structure
ml-assignment/
│
├── data/
│    ├── example_corpus.txt
│    └── large_dataset.txt
│
├── src/
│    ├── ngram_model.py
│    ├── utils.py
│    └── generate.py
│
├── notebooks/
│    └── large_dataset_colab.ipynb  # Optional demo in Colab
│
└── tests/
     └── test_ngram.py
