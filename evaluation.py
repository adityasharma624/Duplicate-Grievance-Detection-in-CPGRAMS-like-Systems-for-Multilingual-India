"""
Evaluation module.

Provides weak evaluation metrics and tools for manual inspection.
Suitable for low-resource scenarios with minimal human annotation.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import config


def compute_cluster_coherence(
    cluster_ids: List[str],
    embeddings: np.ndarray,
    id_to_idx: Dict[str, int]
) -> float:
    """
    Compute cluster coherence as mean pairwise similarity within cluster.
    
    Coherence measures how similar complaints are within a cluster.
    Higher coherence indicates better cluster quality.
    
    Args:
        cluster_ids: List of complaint IDs in cluster
        embeddings: Embedding matrix
        id_to_idx: Mapping from complaint ID to embedding row index
        
    Returns:
        Mean pairwise cosine similarity within cluster
    """
    if len(cluster_ids) < 2:
        return 1.0  # Singleton clusters have perfect coherence
    
    # Get embeddings for cluster
    cluster_embeddings = np.array([
        embeddings[id_to_idx[cid]] for cid in cluster_ids
    ])
    
    # Compute pairwise similarities
    similarity_matrix = cosine_similarity(cluster_embeddings)
    
    # Extract upper triangle (excluding diagonal)
    n = len(cluster_ids)
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            similarities.append(similarity_matrix[i, j])
    
    return float(np.mean(similarities))


def compute_all_cluster_coherences(
    clusters: Dict[int, List[str]],
    embeddings: np.ndarray,
    id_to_idx: Dict[str, int]
) -> Dict[int, float]:
    """
    Compute coherence for all clusters.
    
    Args:
        clusters: Dictionary mapping cluster_id to list of complaint IDs
        embeddings: Embedding matrix
        id_to_idx: Mapping from complaint ID to embedding row index
        
    Returns:
        Dictionary mapping cluster_id to coherence score
    """
    coherences = {}
    for cluster_id, cluster_ids in clusters.items():
        coherences[cluster_id] = compute_cluster_coherence(
            cluster_ids, embeddings, id_to_idx
        )
    return coherences


def sample_clusters_for_inspection(
    clusters: Dict[int, List[str]],
    sample_size: Optional[int] = None
) -> Dict[int, List[str]]:
    """
    Sample clusters for manual inspection.
    
    Prioritizes multi-item clusters (potential duplicates) over singletons.
    
    Args:
        clusters: Dictionary mapping cluster_id to list of complaint IDs
        sample_size: Number of clusters to sample (default: config value)
        
    Returns:
        Dictionary of sampled clusters
    """
    if sample_size is None:
        sample_size = config.EVALUATION_SAMPLE_SIZE
    
    # Separate multi-item and singleton clusters
    multi_item_clusters = {
        cid: ids for cid, ids in clusters.items() if len(ids) > 1
    }
    singleton_clusters = {
        cid: ids for cid, ids in clusters.items() if len(ids) == 1
    }
    
    # Prioritize multi-item clusters
    sampled = {}
    
    # Sample from multi-item clusters first
    if multi_item_clusters:
        n_multi = min(sample_size, len(multi_item_clusters))
        sampled_multi = random.sample(
            list(multi_item_clusters.items()), n_multi
        )
        sampled.update(dict(sampled_multi))
    
    # Fill remaining slots with singletons if needed
    remaining = sample_size - len(sampled)
    if remaining > 0 and singleton_clusters:
        n_singletons = min(remaining, len(singleton_clusters))
        sampled_singletons = random.sample(
            list(singleton_clusters.items()), n_singletons
        )
        sampled.update(dict(sampled_singletons))
    
    return sampled


def export_clusters_for_inspection(
    sampled_clusters: Dict[int, List[str]],
    complaints: List[Dict[str, Any]],
    output_file: Path
):
    """
    Export sampled clusters to JSON for manual inspection.
    
    Args:
        sampled_clusters: Dictionary of clusters to export
        complaints: List of complaint dictionaries
        output_file: Path to output JSON file
    """
    # Create mapping from ID to complaint
    id_to_complaint = {c['id']: c for c in complaints}
    
    # Format clusters with full complaint data
    formatted_clusters = []
    for cluster_id, complaint_ids in sampled_clusters.items():
        cluster_data = {
            'cluster_id': cluster_id,
            'n_complaints': len(complaint_ids),
            'complaints': [
                id_to_complaint[cid] for cid in complaint_ids
            ]
        }
        formatted_clusters.append(cluster_data)
    
    # Write to JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_clusters, f, ensure_ascii=False, indent=2)


def generate_evaluation_report(
    clusters: Dict[int, List[str]],
    cluster_coherences: Dict[int, float],
    duplicate_pairs: List[Dict[str, Any]],
    output_file: Path
):
    """
    Generate comprehensive evaluation report.
    
    Args:
        clusters: Dictionary mapping cluster_id to list of complaint IDs
        cluster_coherences: Dictionary mapping cluster_id to coherence score
        duplicate_pairs: List of duplicate pair dictionaries
        output_file: Path to output report file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append("Duplicate Grievance Detection - Evaluation Report")
    lines.append("=" * 80)
    lines.append("")
    
    # Cluster statistics
    lines.append("Cluster Statistics:")
    lines.append(f"  Total clusters: {len(clusters)}")
    n_singletons = sum(1 for ids in clusters.values() if len(ids) == 1)
    lines.append(f"  Singleton clusters: {n_singletons}")
    lines.append(f"  Multi-item clusters: {len(clusters) - n_singletons}")
    lines.append("")
    
    # Coherence statistics
    if cluster_coherences:
        coherences = list(cluster_coherences.values())
        lines.append("Cluster Coherence Statistics:")
        lines.append(f"  Mean coherence: {np.mean(coherences):.4f}")
        lines.append(f"  Median coherence: {np.median(coherences):.4f}")
        lines.append(f"  Min coherence: {np.min(coherences):.4f}")
        lines.append(f"  Max coherence: {np.max(coherences):.4f}")
        lines.append("")
    
    # Duplicate pairs statistics
    lines.append("Duplicate Pairs Statistics:")
    lines.append(f"  Total pairs found: {len(duplicate_pairs)}")
    if duplicate_pairs:
        similarities = [p['similarity_score'] for p in duplicate_pairs]
        lines.append(f"  Mean similarity: {np.mean(similarities):.4f}")
        lines.append(f"  Median similarity: {np.median(similarities):.4f}")
        lines.append(f"  Min similarity: {np.min(similarities):.4f}")
        lines.append(f"  Max similarity: {np.max(similarities):.4f}")
    lines.append("")
    
    # Top clusters by coherence
    if cluster_coherences:
        lines.append("Top 10 Clusters by Coherence:")
        sorted_clusters = sorted(
            cluster_coherences.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for cluster_id, coherence in sorted_clusters:
            n_complaints = len(clusters[cluster_id])
            lines.append(
                f"  Cluster {cluster_id}: coherence={coherence:.4f}, "
                f"n_complaints={n_complaints}"
            )
    
    # Write report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

