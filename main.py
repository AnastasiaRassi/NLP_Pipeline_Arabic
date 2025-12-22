"""Main entry point for Arabic NLP pipeline."""

import argparse
import sys
from pathlib import Path

from src.config import load_config, set_random_seeds
from src.utils import setup_logging
from src.inference import ArabicNLPPipeline


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Arabic NLP Pipeline: NER and Embedding Generation"
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Arabic text to process (if not provided, reads from stdin; standard input)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: config/config.yaml)"
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedding generation"
    )
    parser.add_argument(
        "--no-pca",
        action="store_true",
        help="Skip PCA dimensionality reduction"
    )
    parser.add_argument(
        "--entities-only",
        action="store_true",
        help="Extract entities only (skip embeddings)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Load configuration
    try:
        config = load_config(args.config)
        set_random_seeds(config["random_seed"])
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get input text
    if args.text:
        input_text = args.text
    elif not sys.stdin.isatty():
        input_text = sys.stdin.read().strip()
    else:
        parser.print_help()
        sys.exit(1)
    
    if not input_text:
        print("Error: No input text provided", file=sys.stderr)
        sys.exit(1)
    
    # Initialize pipeline
    try:
        pipeline = ArabicNLPPipeline(config)
    except Exception as e:
        print(f"Error initializing pipeline: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Process text
    try:
        if args.entities_only:
            entities = pipeline.extract_entities_only(input_text)
            print("\nExtracted Entities:")
            print("-" * 50)
            for entity in entities:
                print(f"  {entity['word']}: {entity['label']}")
        else:
            result = pipeline.predict(
                input_text,
                generate_embeddings=not args.no_embeddings, # so if no embeddings is False, generate_embeddings is True
                apply_pca=not args.no_pca # if no pca, apply pca is False
            )
            
            print("\nExtracted Entities:")
            print("-" * 50)
            for entity in result["entities"]:
                print(f"  {entity['word']}: {entity['label']}")
            
            if "embeddings" in result:
                print(f"\nEmbeddings Shape: {result['embeddings'].shape}")
                if "reduced_embeddings" in result:
                    print(f"Reduced Embeddings Shape: {result['reduced_embeddings'].shape}")
    except Exception as e:
        print(f"Error processing text: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

