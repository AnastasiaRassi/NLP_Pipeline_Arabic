from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA
from src.preprocessing import ArabicPreprocessor
from src.modeling import ArabicNER, EmbeddingGenerator
from src.utils import get_logger

logger = get_logger(__name__)


class ArabicNLPPipeline:    
    def __init__(self, config: Dict):
        self.config = config
        self.preprocessor = ArabicPreprocessor(config)
        self.ner_model = ArabicNER(config)
        self.embedding_generator = EmbeddingGenerator(config)
        logger.info("Arabic NLP pipeline initialized")
    
    def predict(self, text: str, generate_embeddings: bool = True, apply_pca: bool = True) -> Dict:
        """ Returns:
                - entities: List of extracted entities with labels
                - preprocessed_text: Cleaned input text
                - embeddings: Raw embeddings (if generate_embeddings=True)
                - reduced_embeddings: PCA-reduced embeddings (if apply_pca=True)
                - pca_model: PCA model instance (if apply_pca=True)
        """
        logger.info(f"Processing text (length: {len(text)} characters)")
        
        preprocessed_text = self.preprocessor.preprocess(text)
        
        entities = self.ner_model.extract_entities(preprocessed_text)
        
        result = {
            "entities": entities,
            "preprocessed_text": preprocessed_text
        }
        
        if generate_embeddings and entities:  # generate embeddings if set to True
            embeddings, reduced_embeddings, pca_model = self.embedding_generator.generate_embeddings(
                entities, apply_pca=apply_pca
            )
            result["embeddings"] = embeddings
            if apply_pca:
                result["reduced_embeddings"] = reduced_embeddings
                result["pca_model"] = pca_model
        
        return result
    
    def extract_entities_only(self, text: str) -> List[Dict[str, str]]:
        preprocessed_text = self.preprocessor.preprocess(text)
        return self.ner_model.extract_entities(preprocessed_text)


def predict( text: str, config: Dict, generate_embeddings: bool = True, apply_pca: bool = True):
    pipeline = ArabicNLPPipeline(config)
    return pipeline.predict(text, generate_embeddings, apply_pca)

