"""
Main pipeline orchestrator for duplicate grievance detection.

Executes the complete pipeline:
1. Data ingestion
2. Preprocessing
3. Embedding generation
4. Clustering
5. Duplicate pair extraction
6. Evaluation

Usage:
    python main.py --input data/complaints.csv
    python main.py --input data/complaints.json --output-dir results
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import config
import data_ingestion
import preprocessing
import embedding
import clustering
import duplicate_extraction
import evaluation


def run_pipeline(
    input_file: Path,
    output_dir: Optional[Path] = None,
    skip_evaluation: bool = False
) -> Dict[str, Any]:
    """
    Execute the complete duplicate detection pipeline.
    
    Args:
        input_file: Path to input CSV or JSON file
        output_dir: Directory for output files (default: config.OUTPUT_DIR)
        skip_evaluation: If True, skip evaluation step
        
    Returns:
        Dictionary with pipeline results
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Duplicate Grievance Detection Pipeline")
    print("=" * 80)
    print()
    
    # Step 1: Data ingestion
    print("Step 1: Data Ingestion")
    print("-" * 80)
    complaints = data_ingestion.load_complaints(input_file)
    complaints = data_ingestion.validate_complaints(complaints)
    print(f"Loaded {len(complaints)} valid complaints")
    print()
    
    # Step 2: Preprocessing
    print("Step 2: Preprocessing")
    print("-" * 80)
    preprocessed_complaints = preprocessing.preprocess_complaints(complaints)
    print(f"Preprocessed {len(preprocessed_complaints)} complaints")
    print()
    
    # Step 3: Embedding generation
    print("Step 3: Embedding Generation")
    print("-" * 80)
    print(f"Using model: {config.EMBEDDING_MODEL_NAME}")
    embedder = embedding.EmbeddingGenerator()
    embeddings, id_to_idx = embedder.generate_embeddings(
        preprocessed_complaints,
        use_cache=True
    )
    print(f"Generated embeddings of shape {embeddings.shape}")
    print()
    
    # Step 4: Clustering
    print("Step 4: Clustering")
    print("-" * 80)
    print(f"Distance threshold: {config.CLUSTERING_DISTANCE_THRESHOLD}")
    print(f"Linkage method: {config.CLUSTERING_LINKAGE}")
    complaint_ids = [c['id'] for c in preprocessed_complaints]
    clusters = clustering.cluster_complaints(embeddings, complaint_ids)
    cluster_stats = clustering.get_cluster_statistics(clusters)
    print(f"Found {cluster_stats['n_clusters']} clusters")
    print(f"  - Singleton clusters: {cluster_stats['n_singletons']}")
    print(f"  - Multi-item clusters: {cluster_stats['n_multi_item_clusters']}")
    print(f"  - Max cluster size: {cluster_stats['max_cluster_size']}")
    print()
    
    # Step 5: Duplicate pair extraction
    print("Step 5: Duplicate Pair Extraction")
    print("-" * 80)
    print(f"Minimum similarity: {config.MIN_SIMILARITY_SCORE}")
    duplicate_pairs = duplicate_extraction.extract_duplicate_pairs(
        clusters, embeddings, id_to_idx, preprocessed_complaints
    )
    print(f"Extracted {len(duplicate_pairs)} duplicate pairs")
    print()
    
    # Step 6: Evaluation (optional)
    if not skip_evaluation:
        print("Step 6: Evaluation")
        print("-" * 80)
        cluster_coherences = evaluation.compute_all_cluster_coherences(
            clusters, embeddings, id_to_idx
        )
        print(f"Computed coherence for {len(cluster_coherences)} clusters")
        
        # Sample clusters for manual inspection
        sampled_clusters = evaluation.sample_clusters_for_inspection(clusters)
        inspection_file = output_dir / "clusters_for_inspection.json"
        evaluation.export_clusters_for_inspection(
            sampled_clusters, preprocessed_complaints, inspection_file
        )
        print(f"Exported {len(sampled_clusters)} clusters for inspection")
        print(f"  Output: {inspection_file}")
        
        # Generate evaluation report
        report_file = output_dir / "evaluation_report.txt"
        evaluation.generate_evaluation_report(
            clusters, cluster_coherences, duplicate_pairs, report_file
        )
        print(f"Generated evaluation report: {report_file}")
        print()
    
    # Save results
    print("Saving Results")
    print("-" * 80)
    
    # Save duplicate pairs as JSON
    pairs_file = output_dir / "duplicate_pairs.json"
    with open(pairs_file, 'w', encoding='utf-8') as f:
        json.dump(duplicate_pairs, f, ensure_ascii=False, indent=2)
    print(f"Saved duplicate pairs: {pairs_file}")
    
    # Save duplicate pairs as human-readable text
    pairs_text_file = output_dir / "duplicate_pairs.txt"
    pairs_text = duplicate_extraction.format_duplicate_pairs_output(duplicate_pairs)
    with open(pairs_text_file, 'w', encoding='utf-8') as f:
        f.write(pairs_text)
    print(f"Saved duplicate pairs (text): {pairs_text_file}")
    
    # Save cluster assignments
    cluster_assignments = {}
    for cluster_id, complaint_ids in clusters.items():
        for cid in complaint_ids:
            cluster_assignments[cid] = int(cluster_id)
    assignments_file = output_dir / "cluster_assignments.json"
    with open(assignments_file, 'w', encoding='utf-8') as f:
        json.dump(cluster_assignments, f, ensure_ascii=False, indent=2)
    print(f"Saved cluster assignments: {assignments_file}")
    
    print()
    print("=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80)
    
    # Return results summary
    results = {
        'n_complaints': int(len(preprocessed_complaints)),
        'n_clusters': int(cluster_stats['n_clusters']),
        'n_duplicate_pairs': int(len(duplicate_pairs)),
        'output_dir': str(output_dir)
    }
    
    return results


def main():
    """Command-line interface for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Duplicate Grievance Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data/complaints.csv
  python main.py --input data/complaints.json --output-dir results
  python main.py --input data/complaints.csv --skip-evaluation

Note: The input file must be a CSV or JSON file with 'id' and 'text' columns/fields.
        """
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to input CSV or JSON file (required)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help=f'Output directory (default: {config.OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation step for faster execution'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    results = run_pipeline(
        input_file=args.input,
        output_dir=args.output_dir,
        skip_evaluation=args.skip_evaluation
    )
    
    # Print summary
    print("\nSummary:")
    print(f"  Processed {results['n_complaints']} complaints")
    print(f"  Found {results['n_clusters']} clusters")
    print(f"  Identified {results['n_duplicate_pairs']} duplicate pairs")
    print(f"  Results saved to: {results['output_dir']}")


if __name__ == "__main__":
    main()

