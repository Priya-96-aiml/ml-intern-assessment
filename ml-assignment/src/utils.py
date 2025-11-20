# This file is optional.
# You can add any utility functions you need for your implementation here.
import re
import logging
import random
import time
from typing import List, Optional

# Optional imports for advanced NLP
try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except ImportError:
    nltk = None


# ---------------- Text Utilities ---------------- #
def clean_text(
    text: str, 
    remove_stopwords: bool = False, 
    lemmatize: bool = False
) -> str:
    """
    Cleans text for NLP:
    - Lowercases
    - Removes URLs, emojis, and special characters
    - Optionally removes stopwords and lemmatizes
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove emojis and non-alphanumeric characters except apostrophes
    text = re.sub(r"[^\w\s']", " ", text)

    # Normalize spaces
    text = " ".join(text.split())

    # Tokenize
    tokens = text.split()

    # Remove stopwords
    if remove_stopwords and nltk:
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]

    # Lemmatization
    if lemmatize and nltk:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def tokenize(text: str) -> List[str]:
    """Tokenizes cleaned text into words"""
    return text.split() if text else []


# ---------------- Seed Utilities ---------------- #
def set_seed(seed: int = 42):
    """Set seeds for Python, NumPy, and PyTorch"""
    random.seed(seed)
    if np:
        np.random.seed(seed)
    if torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------- Logger Utilities ---------------- #
def get_logger(name: str = "app_logger", colored: bool = True):
    """Returns a logger instance with optional colored output"""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        if colored:
            try:
                from colorama import Fore, Style, init
                init(autoreset=True)
                class ColorFormatter(logging.Formatter):
                    COLORS = {
                        'DEBUG': Fore.CYAN,
                        'INFO': Fore.GREEN,
                        'WARNING': Fore.YELLOW,
                        'ERROR': Fore.RED,
                        'CRITICAL': Fore.MAGENTA
                    }
                    def format(self, record):
                        msg = super().format(record)
                        color = self.COLORS.get(record.levelname, "")
                        return f"{color}{msg}{Style.RESET_ALL}"
                formatter = ColorFormatter(fmt="%(asctime)s | %(levelname)s | %(message)s",
                                           datefmt="%H:%M:%S")
            except ImportError:
                formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s",
                                              datefmt="%H:%M:%S")
        else:
            formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s",
                                          datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    return logger


# ---------------- Decorators ---------------- #
def timing(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIMING] {func.__name__} executed in {end-start:.4f} seconds")
        return result
    return wrapper
