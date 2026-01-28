# Arabic NLP Pipeline

Production-ready Arabic text processing pipeline for Named Entity Recognition (NER) and embedding generation. Provides an end-to-end solution for extracting and vectorizing named entities from Arabic text.

## Overview

This system performs Arabic text preprocessing, entity extraction using transformer-based models, and generates embeddings for downstream NLP tasks. The pipeline is designed with modularity and configurability as core principles.

## Problem Statement

Arabic NLP presents unique challenges including complex morphology, diacritical marks, and character normalization requirements. This system addresses entity extraction and representation for Arabic text, providing a configurable pipeline suitable for integration into production systems.

## Approach

- **Text Preprocessing**: Arabic-specific normalization, diacritics removal, and stopword filtering
- **Named Entity Recognition**: Transformer-based model (`hatmimoha/arabic-ner`) for entity extraction
- **Embedding Generation**: Word2Vec embeddings with optional PCA dimensionality reduction

## Installation

### Requirements

- Python 3.8+
- PyTorch (automatically installed with transformers)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for preprocessing)
python setup_nltk.py
```

### Configuration

1. Copy `config/config.yaml` and customize settings as needed
2. Set `HUGGINGFACE_TOKEN` environment variable if using private models:
   ```bash
   export HUGGINGFACE_TOKEN=your_token_here
   ```

## Usage

### Command Line

```bash
# Basic usage
python main.py "الرئيس اللبناني يعقد اجتماعاً لمناقشة التحديات الاقتصادية."

# Extract entities only (skip embeddings)
python main.py "النص العربي" --entities-only

# Disable PCA
python main.py "النص العربي" --no-pca

# From stdin
echo "النص العربي" | python main.py
```

### Python API

```python
from src.config import load_config, set_random_seeds
from src.inference import ArabicNLPPipeline

# Load configuration
config = load_config()
set_random_seeds(config["random_seed"])

# Initialize pipeline
pipeline = ArabicNLPPipeline(config)

# Process text
result = pipeline.predict("الرئيس اللبناني يعقد اجتماعاً", generate_embeddings=True)

# Access results
entities = result["entities"]  # List of {word, label} dicts
embeddings = result["embeddings"]  # NumPy array
reduced = result["reduced_embeddings"]  # PCA-reduced embeddings
```

### Quick Start Example

See `example.py` for a complete working example.

## Architecture

```
src/
  ├── preprocessing.py    # Arabic text preprocessing
  ├── modeling.py         # NER and embedding models
  ├── inference.py        # End-to-end pipeline
  ├── config.py           # Configuration management
  └── utils.py            # Logging and utilities

config/
  └── config.yaml         # System configuration

main.py                   # CLI entry point
example.py                # Usage examples
```

## Configuration

All system parameters are configurable via `config/config.yaml`:

- **Model settings**: NER model name, aggregation strategy
- **Preprocessing**: Stopword lists, normalization rules
- **Embeddings**: Word2Vec parameters, PCA dimensions
- **Reproducibility**: Random seed settings

## Results

The system extracts named entities from Arabic text and generates vector representations suitable for similarity search, clustering, or as input to downstream ML models. Entity labels follow the NER model's output schema.

### Performance Considerations

- Initial model loading may take 30-60 seconds (models are cached)
- Processing time: ~0.5-2 seconds per sentence depending on length
- Memory usage: ~2-4 GB for model and embeddings

## Limitations

1. **Word2Vec Training**: Current implementation trains Word2Vec on single-sentence entity lists. For production use, consider pre-trained Arabic embeddings or training on larger corpora.

2. **Model Dependency**: Performance depends on the underlying NER model (`hatmimoha/arabic-ner`). Model quality varies by domain and dialect.

3. **No Evaluation Framework**: This system focuses on inference. For evaluation metrics (precision, recall, F1), provide labeled test data.

4. **Dialect Coverage**: Standard Arabic focus; dialectal variations may reduce accuracy.

## Future Improvements

- Integration with pre-trained Arabic word embeddings (e.g., AraVec, fastText)
- Support for multiple NER models with model comparison
- Evaluation framework with standard metrics
- Batch processing capabilities
- API server implementation (FastAPI/Flask)
- Support for additional Arabic dialects

## License

Open source. See LICENSE file for details.
