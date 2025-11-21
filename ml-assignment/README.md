# Trigram Language Model

This directory contains the core implementation files for the Trigram Language Model, which predicts the next word in a sequence using two preceding words. The model is trained on a text corpus and generates new text based on learned trigram probabilities.

## How to Run 

1 Clone the Repository
      git clone https://github.com/VarshiniAG/ml-intern-assessment.git
     cd ml-intern-assessment

2️ Prepare the Environment

Install required dependencies (if any new ones are added later):

     pip install -r requirements.txt

3️ Run the Trigram Model
python generate.py

4️ Output

The script will print generated text based on the trained trigram language model.

## Design Choices

1. Model Selection

The project uses a Trigram Language Model, choosing the next word based on the previous two words.
This method balances contextual understanding and computational simplicity—more informative than bigram models and less complex than full neural language models.

2. Tokenization Strategy

Converted text to lowercase

Removed punctuation and special characters

Split words into tokens based on whitespace

This ensures a clean and consistent dataset, reducing noise and improving language modeling quality.

3. Data Structure for Trigrams

A dictionary of bigram → candidate words mapping was used:

trigrams[(word1, word2)] = [possible_next_words]


This structure allows:

Fast lookup while generating text

Storage of multiple candidate words with equal probability

4. Text Generation Approach

Started from a random bigram to introduce variability

Iteratively selected the next word using random choice from available candidates

Maximum word length is controlled (max_words=20) to prevent infinite loops

This results in realistic yet diverse sentence formation.

5. Edge Case Handling

The model returns:

"" for empty input text

Original token list if trigrams cannot be formed

Graceful stopping when no continuation exists

This improves reliability and prevents runtime errors.

6. Code Organization

The project is modularized into:

trigram_model.py – core logic

generate.py – entry script

data/ – training corpus

utils/ – helper functions (logger, seed setup, text cleaning)

This structure supports maintainability and potential future extensions.

7. Future Improvements

Add smoothing (Laplace smoothing for unseen n-grams)

Add probability-based sampling instead of uniform random selection

Integrate evaluation metrics like perplexity

Create Streamlit UI for interactive text generation



Structure

ml-assignment/

│

├── data/

│   └── example_corpus.txt

│

├── src/

│   ├── generate.py

│   ├── ngram_model.py

│   └── utils.py

│

├── tests/

│   └── test_ngram.py

│

├── README.md

├── evaluation.md

├── assignment.md

├── quick_start.md

├── requirements.txt

└── .gitignore

