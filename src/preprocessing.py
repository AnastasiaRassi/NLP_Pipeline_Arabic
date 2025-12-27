"""Arabic text preprocessing module."""

import os
import re
from typing import Dict, List
import nltk
from nltk.corpus import stopwords

from src.utils import get_logger

logger = get_logger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.warning(
        'NLTK stopwords not found. Run: python setup_nltk.py or '
        'python -c "import nltk; nltk.download(\'stopwords\')"'
    )


class ArabicPreprocessor:    
    def __init__(self, config: Dict):
        self.config = config
        self._load_stopwords()
        self._setup_normalization()
        self._compile_patterns()
    
    def _load_stopwords(self) -> None:
        """Loads and configures Arabic stopwords. Combines:
        - NLTK's built-in Arabic stopword list
        - stopwords from `stop_words_arabic.txt` at the project root, 
            source: https://github.com/mohataher/arabic-stop-words
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        custom_path = os.path.join(project_root, "stop_words_arabic.txt")

        arabic_stopwords = set()
        if os.path.exists(custom_path):
            with open(custom_path, encoding="utf-8") as f:
                arabic_stopwords = {
                    line.strip() for line in f
                    if line.strip() and not line.lstrip().startswith("#")
                }
        else:
            logger.warning(f"Custom stopword file not found at: {custom_path}")

        # Load NLTK Arabic stopwords 
        nltk_stopwords = set()
        try:
            nltk_stopwords = set(stopwords.words("arabic"))
        except LookupError:
            logger.warning(
                "NLTK Arabic stopwords not available. "
                "Run: python setup_nltk.py to download them."
            )

        self.arabic_stopwords = arabic_stopwords.union(nltk_stopwords)

        logger.debug(
            f"Loaded {len(arabic_stopwords)} custom stopwords and "
            f"{len(nltk_stopwords)} NLTK stopwords "
            f"(total unique: {len(self.arabic_stopwords)})"
        )
    
    def _setup_normalization(self) -> None:
        self.normalization_map = dict(
            self.config.get("preprocessing", {}).get("normalization_map", {})
        )

    def _compile_patterns(self) -> None:
        self.arabic_diacritics = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
        self.punctuation_pattern = re.compile(r'[^\w\s]')
    
    def preprocess(self, text: str) -> str:
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.arabic_stopwords]
        
        text = " ".join(tokens)
        text = re.sub(self.arabic_diacritics, '', text)
        
        text = re.sub(self.punctuation_pattern, '', text)
        
        for key, value in self.normalization_map.items():
            text = text.replace(key, value)
        
        logger.debug(f"Preprocessed text length: {len(text)} characters")
        return text


def preprocess_arabic(text: str, config: Dict) -> str:
    preprocessor = ArabicPreprocessor(config)
    return preprocessor.preprocess(text)

