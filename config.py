"""
Configuration module for duplicate grievance detection system.

Centralizes all system parameters to ensure reproducibility.
All design choices are documented here.
"""

from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROJECT_ROOT / "cache"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
EMBEDDING_CACHE_FILE = CACHE_DIR / "embeddings.npy"
EMBEDDING_IDS_CACHE_FILE = CACHE_DIR / "embedding_ids.json"

CLUSTERING_ALGORITHM = "agglomerative"
CLUSTERING_DISTANCE_THRESHOLD = 0.3
CLUSTERING_LINKAGE = "average"

MIN_SIMILARITY_SCORE = 0.7
TOP_K_PAIRS = 100

NORMALIZE_WHITESPACE = True
LOWERCASE = False
REMOVE_SPECIAL_CHARS = False

EVALUATION_SAMPLE_SIZE = 20
EVALUATION_OUTPUT_DIR = OUTPUT_DIR / "evaluation"

SUPPORTED_INPUT_FORMATS = [".csv", ".json"]
TEXT_COLUMN_NAME = "text"
ID_COLUMN_NAME = "id"

