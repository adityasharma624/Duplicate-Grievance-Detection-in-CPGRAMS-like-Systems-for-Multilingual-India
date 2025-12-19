"""
Configuration module for duplicate grievance detection system.

Centralizes all system parameters to ensure reproducibility.
All design choices are documented here.
"""

from pathlib import Path
from typing import Optional

# Base paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROJECT_ROOT / "cache"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Embedding configuration
# Model choice: paraphrase-multilingual-MiniLM-L12-v2
# Rationale: 
# - Supports 50+ languages including all target languages (English, Hindi, Bengali, Marathi, Tamil, Telugu)
# - Lightweight (80MB) suitable for CPU-only execution
# - Produces 384-dimensional embeddings
# - No fine-tuning required, pretrained on multilingual paraphrase data
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
EMBEDDING_CACHE_FILE = CACHE_DIR / "embeddings.npy"
EMBEDDING_IDS_CACHE_FILE = CACHE_DIR / "embedding_ids.json"

# Clustering configuration
# Algorithm choice: Agglomerative Clustering with cosine distance
# Rationale:
# - Deterministic and interpretable (important for academic work)
# - Scales reasonably on CPU (O(n^2 log n) for linkage, O(n^2) memory)
# - Cosine distance is appropriate for normalized embeddings
# - Allows control over number of clusters via distance threshold
CLUSTERING_ALGORITHM = "agglomerative"
CLUSTERING_DISTANCE_THRESHOLD = 0.3  # Cosine distance threshold for merging clusters
# Lower threshold = more clusters (stricter duplicate detection)
# Higher threshold = fewer clusters (more lenient duplicate detection)
CLUSTERING_LINKAGE = "average"  # Average linkage balances single and complete linkage

# Duplicate extraction configuration
MIN_SIMILARITY_SCORE = 0.7  # Minimum cosine similarity to consider as duplicate pair
TOP_K_PAIRS = 100  # Number of top duplicate pairs to output

# Preprocessing configuration
# Minimal normalization to preserve semantic content
# Rationale: Aggressive normalization may remove important linguistic cues
NORMALIZE_WHITESPACE = True
LOWERCASE = False  # Preserve case for proper nouns and acronyms
REMOVE_SPECIAL_CHARS = False  # Preserve punctuation for context

# Evaluation configuration
EVALUATION_SAMPLE_SIZE = 20  # Number of clusters to sample for manual inspection
EVALUATION_OUTPUT_DIR = OUTPUT_DIR / "evaluation"

# Data format expectations
SUPPORTED_INPUT_FORMATS = [".csv", ".json"]
TEXT_COLUMN_NAME = "text"  # Expected column name for complaint text
ID_COLUMN_NAME = "id"  # Expected column name for complaint ID

