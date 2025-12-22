"""Example usage of the Arabic NLP pipeline."""

from src.config import load_config, set_random_seeds
from src.utils import setup_logging
from src.inference import ArabicNLPPipeline


def main():
    # Setup logging
    setup_logging(level="INFO")
    
    # Load configuration
    config = load_config()
    set_random_seeds(config["random_seed"])
    
    # Initialize pipeline
    pipeline = ArabicNLPPipeline(config)
    
    # Example Arabic text
    sample_text = "الرئيس اللبناني يعقد اجتماعاً لمناقشة التحديات الاقتصادية."
    
    print(f"Input text: {sample_text}\n")
    
    # Process through pipeline
    result = pipeline.predict(sample_text, generate_embeddings=True, apply_pca=True)
    
    # Display results
    print("Extracted Entities:")
    print("-" * 50)
    for entity in result["entities"]:
        print(f"  {entity['word']}: {entity['label']}")
    
    if "embeddings" in result:
        print(f"\nEmbeddings generated: {result['embeddings'].shape}")
        if "reduced_embeddings" in result:
            print(f"PCA reduced shape: {result['reduced_embeddings'].shape}")


if __name__ == "__main__":
    main()

