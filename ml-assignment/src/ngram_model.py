
import random
import re
from collections import defaultdict
from typing import List, Tuple, Optional

class TrigramModel:
    def __init__(self, seed: Optional[int] = None):
        """
        Trigram model for text generation.
        :param seed: Optional random seed for reproducibility.
        """
        self.trigrams: dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.tokens: List[str] = []
        if seed is not None:
            random.seed(seed)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize input text:
        - Lowercase
        - Remove punctuation
        - Split on whitespace
        """
        if not text:
            return []
        text = text.lower()
        text = re.sub(r"[^\w\s']", "", text)  # keep apostrophes
        tokens = text.split()
        return tokens

    def fit(self, text: str):
        """
        Build trigram dictionary from text.
        """
        self.tokens = self._tokenize(text)
        self.trigrams.clear()

        if len(self.tokens) < 3:
            return  # Not enough tokens for trigram

        for i in range(len(self.tokens) - 2):
            key = (self.tokens[i], self.tokens[i + 1])
            next_word = self.tokens[i + 2]
            self.trigrams[key].append(next_word)

    def generate(self, max_words: int = 20, start_bigram: Optional[Tuple[str, str]] = None) -> str:
        """
        Generate text based on trigrams.
        :param max_words: Maximum number of words to generate
        :param start_bigram: Optional starting bigram tuple
        :return: Generated text string
        """
        if not self.tokens:
            return ""

        if not self.trigrams:
            return " ".join(self.tokens)

        if start_bigram and start_bigram in self.trigrams:
            current_bigram = start_bigram
        else:
            current_bigram = random.choice(list(self.trigrams.keys()))

        output_words = [current_bigram[0], current_bigram[1]]

        for _ in range(max_words):
            next_words = self.trigrams.get(current_bigram)
            if not next_words:
                break
            next_word = random.choice(next_words)
            output_words.append(next_word)
            current_bigram = (current_bigram[1], next_word)

        return " ".join(output_words)

