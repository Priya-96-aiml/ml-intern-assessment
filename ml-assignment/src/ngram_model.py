import random
from utils import clean_text, tokenize

class TrigramModel:
    def __init__(self):
        """
        Initializes the trigram model.
        """
        self.trigrams = {}
        self.start_tokens = ('<s>', '<s>')
        self.has_data = False  # Used for empty-text handling

    def fit(self, text):
        """
        Train the trigram model on the given text.
        """
        text = clean_text(text)
        words = tokenize(text)

        # Handle empty or too-short input
        if len(words) == 0:
            self.trigrams = {}
            self.has_data = False
            return

        self.has_data = True

        # Add start and end tokens
        words = ['<s>', '<s>'] + words + ['</s>']

        # Build trigram counts
        for i in range(len(words) - 2):
            w1, w2, w3 = words[i], words[i+1], words[i+2]
            key = (w1, w2)
            if key not in self.trigrams:
                self.trigrams[key] = []
            self.trigrams[key].append(w3)

    def generate(self, max_length=50):
        """
        Generate text using the trained trigram model.
        """
        if not self.has_data:
            return ""

        w1, w2 = self.start_tokens
        output = []

        for _ in range(max_length):
            key = (w1, w2)

            # No possible next word
            if key not in self.trigrams:
                break

            next_word = random.choice(self.trigrams[key])

            if next_word == '</s>':
                break

            output.append(next_word)
            w1, w2 = w2, next_word

        return " ".join(output)
