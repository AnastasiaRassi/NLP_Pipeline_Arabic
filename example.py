from src.config import load_config, set_random_seeds
from src.utils import setup_logging
from src.inference import ArabicNLPPipeline


def main():
    setup_logging(level="INFO")
    
    config = load_config()
    set_random_seeds(config["random_seed"])
    
    pipeline = ArabicNLPPipeline(config)
    
    sample_text = "الرئيس اللبناني يعقد اجتماعاً لمناقشة التحديات الاقتصادية."
    
    print(f"Input text: {sample_text}\n")
    
    result = pipeline.predict(sample_text, generate_embeddings=True, apply_pca=True)
    
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

