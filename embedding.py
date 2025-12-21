"""
Embedding module.

Generates multilingual sentence embeddings using pretrained models.
Implements disk caching to avoid recomputation.
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import config


class EmbeddingGenerator:
    """
    Generates and caches sentence embeddings.
    
    Uses pretrained multilingual sentence transformer model.
    Caches embeddings to disk to support incremental processing.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of sentence transformer model.
                       Defaults to config.EMBEDDING_MODEL_NAME
        """
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.model: Optional[SentenceTransformer] = None
        self.cache_file = config.EMBEDDING_CACHE_FILE
        self.ids_cache_file = config.EMBEDDING_IDS_CACHE_FILE
    
    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
    
    def _load_cache(self) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """
        Load embeddings and IDs from cache if available.
        
        Returns:
            Tuple of (embeddings array, list of complaint IDs)
        """
        if not self.cache_file.exists() or not self.ids_cache_file.exists():
            return None, None
        
        try:
            embeddings = np.load(self.cache_file)
            with open(self.ids_cache_file, 'r', encoding='utf-8') as f:
                cached_ids = json.load(f)
            return embeddings, cached_ids
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
            return None, None
    
    def _save_cache(self, embeddings: np.ndarray, complaint_ids: List[str]):
        """
        Save embeddings and IDs to cache.
        
        Args:
            embeddings: Embedding matrix
            complaint_ids: List of complaint IDs in same order as embeddings
        """
        np.save(self.cache_file, embeddings)
        with open(self.ids_cache_file, 'w', encoding='utf-8') as f:
            json.dump(complaint_ids, f, ensure_ascii=False, indent=2)
    
    def generate_embeddings(
        self, 
        complaints: List[Dict[str, Any]],
        use_cache: bool = True
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Generate embeddings for complaints with caching support.
        
        Strategy:
        1. Load existing cache if available
        2. Identify complaints not in cache
        3. Generate embeddings only for new complaints
        4. Merge and save updated cache
        
        Args:
            complaints: List of complaint dictionaries with 'id' and 'text'
            use_cache: Whether to use disk cache (default: True)
            
        Returns:
            Tuple of (embedding matrix, mapping from complaint ID to row index)
        """
        complaint_ids = [c['id'] for c in complaints]
        texts = [c['text'] for c in complaints]
        
        cached_embeddings = None
        cached_ids = None
        if use_cache:
            cached_embeddings, cached_ids = self._load_cache()
        
        if cached_embeddings is not None and cached_ids is not None:
            cached_id_to_idx = {cid: idx for idx, cid in enumerate(cached_ids)}
            
            new_complaint_indices = []
            new_texts = []
            for idx, cid in enumerate(complaint_ids):
                if cid not in cached_id_to_idx:
                    new_complaint_indices.append(idx)
                    new_texts.append(texts[idx])
            
            if new_texts:
                self._load_model()
                new_embeddings = self.model.encode(
                    new_texts,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                
                all_embeddings = np.vstack([cached_embeddings, new_embeddings])
                all_ids = cached_ids + [complaint_ids[i] for i in new_complaint_indices]
            else:
                all_embeddings = cached_embeddings
                all_ids = cached_ids
        else:
            self._load_model()
            all_embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            all_ids = complaint_ids
        
        if use_cache:
            self._save_cache(all_embeddings, all_ids)
        
        id_to_idx = {cid: idx for idx, cid in enumerate(all_ids)}
        
        requested_embeddings = np.array([
            all_embeddings[id_to_idx[cid]] for cid in complaint_ids
        ])
        
        requested_id_to_idx = {cid: idx for idx, cid in enumerate(complaint_ids)}
        
        return requested_embeddings, requested_id_to_idx

