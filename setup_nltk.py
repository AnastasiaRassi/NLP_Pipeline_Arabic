"""Setup script to download required NLTK data."""

import nltk

def setup_nltk_data():
    """Download required NLTK corpora and data."""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        print("NLTK data downloaded successfully.")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        raise

if __name__ == "__main__":
    setup_nltk_data()

