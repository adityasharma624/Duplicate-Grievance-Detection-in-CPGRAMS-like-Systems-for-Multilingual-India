"""
Clustering module.

Groups complaints into clusters of potential duplicates.
Uses CPU-efficient algorithms suitable for academic reproducibility.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from typing import List, Dict, Any, Tuple
import config


def cluster_complaints(
    embeddings: np.ndarray,
    complaint_ids: List[str]
) -> Dict[int, List[str]]:
    """
    Cluster complaints using agglomerative clustering.
    
    Algorithm choice: Agglomerative Clustering with cosine distance
    Rationale:
    - Deterministic and interpretable (important for academic work)
    - No random initialization, reproducible results
    - Scales reasonably on CPU (O(n^2 log n) for linkage computation)
    - Cosine distance is appropriate for normalized embeddings
    - Distance threshold provides intuitive control over cluster granularity
    
    Args:
        embeddings: Numpy array of shape (n_complaints, embedding_dim)
        complaint_ids: List of complaint IDs in same order as embeddings
        
    Returns:
        Dictionary mapping cluster_id to list of complaint IDs in that cluster
    """
    n_complaints = len(embeddings)
    
    if n_complaints == 0:
        return {}
    
    if n_complaints == 1:
        # Single complaint forms its own cluster
        return {0: complaint_ids}
    
    # Normalize embeddings to unit length for cosine distance
    # This ensures cosine distance = 1 - cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    norms = np.where(norms == 0, 1, norms)
    normalized_embeddings = embeddings / norms
    
    # Compute pairwise cosine distances
    # cosine_distances returns 1 - cosine_similarity
    # Range: [0, 2] where 0 = identical, 2 = opposite
    distance_matrix = cosine_distances(normalized_embeddings)
    
    # Apply agglomerative clustering
    # distance_threshold: maximum distance for merging clusters
    # linkage: 'average' balances between 'complete' (strict) and 'single' (lenient)
    clustering = AgglomerativeClustering(
        n_clusters=None,  # Let distance_threshold determine number of clusters
        distance_threshold=config.CLUSTERING_DISTANCE_THRESHOLD,
        linkage=config.CLUSTERING_LINKAGE,
        metric='precomputed'  # Use precomputed distance matrix
    )
    
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    # Group complaint IDs by cluster
    clusters = {}
    for complaint_id, cluster_id in zip(complaint_ids, cluster_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(complaint_id)
    
    return clusters


def get_cluster_statistics(clusters: Dict[int, List[str]]) -> Dict[str, Any]:
    """
    Compute statistics about clustering results.
    
    Args:
        clusters: Dictionary mapping cluster_id to list of complaint IDs
        
    Returns:
        Dictionary with clustering statistics
    """
    cluster_sizes = [len(complaint_ids) for complaint_ids in clusters.values()]
    
    stats = {
        'n_clusters': len(clusters),
        'n_singletons': sum(1 for size in cluster_sizes if size == 1),
        'n_multi_item_clusters': sum(1 for size in cluster_sizes if size > 1),
        'max_cluster_size': int(max(cluster_sizes)) if cluster_sizes else 0,
        'mean_cluster_size': float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
        'median_cluster_size': float(np.median(cluster_sizes)) if cluster_sizes else 0.0
    }
    
    return stats

