# NER and embedding generation models.

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from src.utils import get_logger
from src.config import set_random_seeds
import os
from dotenv import load_dotenv

load_dotenv() 

logger = get_logger(__name__)


class ArabicNER:
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config["model"]["name"]
        self.token= os.getenv("HUGGINGFACE_TOKEN")
        self.aggregation_strategy = config["model"]["aggregation_strategy"]
        self._load_model()
    
    def _load_model(self) -> None:   # load NER model and tokenizer
        logger.info(f"Loading NER model: {self.model_name}")
        try:
            tokenizer_kwargs = {"token": self.token} if self.token else {}
            model_kwargs = {"token": self.token} if self.token else {}
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name, **model_kwargs
            )
            self.nlp_ner = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy=self.aggregation_strategy
            )
            logger.info("NER model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NER model: {e}")
            raise
    
    def extract_entities(self, text: str):
        if not text:
            logger.warning("Empty text provided for NER")
            return []
        
        try:
            ner_results = self.nlp_ner(text)
            entities = self._clean_entities(ner_results)
            logger.info(f"Extracted {len(entities)} entities")
            return entities
        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            raise
    
    def _clean_entities(self, ner_results: List[Dict]):
        cleaned_results = []
        
        for entity in ner_results:
            word = entity["word"].replace("##", "")
            
            if cleaned_results and word.startswith(" "):     
                cleaned_results[-1]["word"] += word 
            else:
                cleaned_results.append({
                    "word": word,
                    "label": entity["entity_group"]
                })
        
        return cleaned_results


class EmbeddingGenerator:    
    def __init__(self, config: Dict):
        self.config = config
        self.word2vec_config = config["embeddings"]["word2vec"]
        self.pca_config = config["embeddings"]["pca"]
        set_random_seeds(self.word2vec_config.get("seed", 42))
    
    def generate_embeddings(
        self,
        entities: List[Dict[str, str]],
        apply_pca: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[PCA]]:
        if not entities:
            logger.warning("No entities provided for embedding generation")
            return np.array([]), None, None
        
        entity_words = [entity["word"] for entity in entities]
        logger.info(f"Generating embeddings for {len(entity_words)} entities")
        
        # Train Word2Vec model (Training on a single sentence is not ideal, but maintained for compatibility)
        model = Word2Vec(
            sentences=[entity_words],
            vector_size=self.word2vec_config["vector_size"],
            window=self.word2vec_config["window"],
            min_count=self.word2vec_config["min_count"],
            workers=self.word2vec_config["workers"],
            seed=self.word2vec_config["seed"]
        )
        
        word_vectors = []
        valid_entities = []
        for entity in entity_words:
            if entity in model.wv:
                word_vectors.append(model.wv[entity])
                valid_entities.append(entity)
        
        if len(word_vectors) == 0:
            logger.warning("No valid word vectors generated")
            return np.array([]), None, None
        
        word_vectors = np.array(word_vectors)
        logger.info(f"Generated {len(word_vectors)} valid word vectors")
        
        reduced_vectors = None
        pca_model = None
        
        if apply_pca and len(word_vectors) > 1:
            n_components = min(
                self.pca_config["n_components"],
                len(word_vectors) - 1
            )
            pca_model = PCA(n_components=n_components)
            reduced_vectors = pca_model.fit_transform(word_vectors)
            logger.info(f"Applied PCA: {word_vectors.shape[1]} -> {n_components} dimensions")
        
        return word_vectors, reduced_vectors, pca_model

