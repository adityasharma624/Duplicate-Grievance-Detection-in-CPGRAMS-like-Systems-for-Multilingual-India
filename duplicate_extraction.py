"""
Duplicate pair extraction module.

Extracts potential duplicate pairs from clusters and assigns similarity scores.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import config


def extract_duplicate_pairs(
    clusters: Dict[int, List[str]],
    embeddings: np.ndarray,
    id_to_idx: Dict[str, int],
    complaints: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract duplicate pairs from clusters.
    
    Strategy:
    1. Within each cluster, generate all pairs of complaints
    2. Compute cosine similarity for each pair
    3. Filter by minimum similarity threshold
    4. Sort by similarity score (descending)
    5. Return top-k pairs
    
    Args:
        clusters: Dictionary mapping cluster_id to list of complaint IDs
        embeddings: Embedding matrix
        id_to_idx: Mapping from complaint ID to embedding row index
        complaints: List of complaint dictionaries (for metadata)
        
    Returns:
        List of duplicate pair dictionaries with similarity scores
    """
    duplicate_pairs = []
    
    # Process each cluster
    for cluster_id, complaint_ids in clusters.items():
        # Skip singleton clusters (no pairs possible)
        if len(complaint_ids) < 2:
            continue
        
        # Generate all pairs within cluster
        for i in range(len(complaint_ids)):
            for j in range(i + 1, len(complaint_ids)):
                id1, id2 = complaint_ids[i], complaint_ids[j]
                
                # Get embeddings
                idx1 = id_to_idx[id1]
                idx2 = id_to_idx[id2]
                emb1 = embeddings[idx1:idx1+1]  # Keep 2D shape
                emb2 = embeddings[idx2:idx2+1]
                
                # Compute cosine similarity
                similarity = cosine_similarity(emb1, emb2)[0, 0]
                
                # Filter by minimum similarity threshold
                if similarity >= config.MIN_SIMILARITY_SCORE:
                    # Find complaint metadata
                    complaint1 = next(c for c in complaints if c['id'] == id1)
                    complaint2 = next(c for c in complaints if c['id'] == id2)
                    
                    pair = {
                        'complaint_id_1': id1,
                        'complaint_id_2': id2,
                        'similarity_score': float(similarity),
                        'cluster_id': int(cluster_id),  # Ensure native Python int for JSON serialization
                        'text_1': complaint1['text'],
                        'text_2': complaint2['text']
                    }
                    duplicate_pairs.append(pair)
    
    # Sort by similarity score (descending)
    duplicate_pairs.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Return top-k pairs
    return duplicate_pairs[:config.TOP_K_PAIRS]


def format_duplicate_pairs_output(duplicate_pairs: List[Dict[str, Any]]) -> str:
    """
    Format duplicate pairs for human-readable output.
    
    Args:
        duplicate_pairs: List of duplicate pair dictionaries
        
    Returns:
        Formatted string representation
    """
    lines = []
    lines.append(f"Found {len(duplicate_pairs)} duplicate pairs\n")
    lines.append("=" * 80)
    
    for idx, pair in enumerate(duplicate_pairs, 1):
        lines.append(f"\nPair #{idx}")
        lines.append(f"Similarity Score: {pair['similarity_score']:.4f}")
        lines.append(f"Cluster ID: {pair['cluster_id']}")
        lines.append(f"Complaint ID 1: {pair['complaint_id_1']}")
        lines.append(f"Complaint ID 2: {pair['complaint_id_2']}")
        lines.append(f"\nText 1: {pair['text_1'][:200]}...")
        lines.append(f"Text 2: {pair['text_2'][:200]}...")
        lines.append("-" * 80)
    
    return "\n".join(lines)

