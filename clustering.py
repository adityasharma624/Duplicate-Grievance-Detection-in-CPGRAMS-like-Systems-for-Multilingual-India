import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from typing import List, Dict, Any, Tuple
import config


def cluster_complaints(
    embeddings: np.ndarray,
    complaint_ids: List[str]
) -> Dict[int, List[str]]:
    n_complaints = len(embeddings)
    
    if n_complaints == 0:
        return {}
    
    if n_complaints == 1:
        return {0: complaint_ids}
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized_embeddings = embeddings / norms
    
    distance_matrix = cosine_distances(normalized_embeddings)
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=config.CLUSTERING_DISTANCE_THRESHOLD,
        linkage=config.CLUSTERING_LINKAGE,
        metric='precomputed'
    )
    
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    clusters = {}
    for complaint_id, cluster_id in zip(complaint_ids, cluster_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(complaint_id)
    
    return clusters


def get_cluster_statistics(clusters: Dict[int, List[str]]) -> Dict[str, Any]:
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

