# This file is optional.
# You can add any utility functions you need for your implementation here.
import re

def clean_text(text):
    """
    Lowercase and remove punctuation.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def tokenize(text):
    """
    Split cleaned text into word tokens.
    """
    return text.split()
