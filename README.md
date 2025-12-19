# Duplicate Grievance Detection in CPGRAMS-like Systems for Multilingual India

A research-grade NLP system for detecting duplicate complaints in multilingual grievance management systems. This system is designed for academic submission and focuses on reproducibility, modularity, and CPU-only execution.

## Overview

This system identifies duplicate complaints that refer to the same underlying issue type and entity, regardless of linguistic variation or phrasing. It supports six languages: English, Hindi, Bengali, Marathi, Tamil, and Telugu.

### Key Features

- **Pretrained Models Only**: Uses multilingual sentence embeddings without fine-tuning
- **CPU-Optimized**: Designed for MacBook Air (Apple Silicon) CPU-only execution
- **Modular Architecture**: One module per pipeline stage for clarity and maintainability
- **Reproducible**: Deterministic algorithms with comprehensive configuration
- **Academic-Grade**: Code suitable for submission alongside research papers

## System Architecture

The pipeline consists of six stages:

1. **Data Ingestion**: Loads complaints from CSV or JSON files
2. **Preprocessing**: Minimal text normalization preserving semantic content
3. **Embedding**: Generates multilingual sentence embeddings with disk caching
4. **Clustering**: Groups complaints using agglomerative clustering
5. **Duplicate Extraction**: Identifies duplicate pairs within clusters
6. **Evaluation**: Computes metrics and exports samples for manual inspection

## Installation

### Prerequisites

- Python 3.8 or higher
- macOS (tested on Apple Silicon)

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd Duplicate-Grievance-Detection-in-CPGRAMS-like-Systems-for-Multilingual-India
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import sentence_transformers; print('Installation successful')"
```

## Usage

### Basic Usage

Run the pipeline on a CSV or JSON file:

```bash
python main.py --input data/complaints.csv
```

### Advanced Usage

Specify custom output directory:

```bash
python main.py --input data/complaints.json --output-dir results
```

Skip evaluation for faster execution:

```bash
python main.py --input data/complaints.csv --skip-evaluation
```

### Input Format

The system expects input files in CSV or JSON format with the following structure:

**CSV Format:**
```csv
id,text
1,"Complaint text in any supported language..."
2,"Another complaint..."
```

**JSON Format:**
```json
[
  {"id": "1", "text": "Complaint text in any supported language..."},
  {"id": "2", "text": "Another complaint..."}
]
```

Required fields:
- `id`: Unique identifier for each complaint
- `text`: Complaint text (can be in any supported language or mixed)

Optional: Additional metadata columns will be preserved in outputs.

## Configuration

System parameters can be adjusted in `config.py`:

### Embedding Configuration
- `EMBEDDING_MODEL_NAME`: Pretrained model (default: `paraphrase-multilingual-MiniLM-L12-v2`)
- `EMBEDDING_DIM`: Embedding dimensionality (384 for default model)

### Clustering Configuration
- `CLUSTERING_DISTANCE_THRESHOLD`: Cosine distance threshold for merging clusters (default: 0.3)
  - Lower values = more clusters (stricter duplicate detection)
  - Higher values = fewer clusters (more lenient duplicate detection)
- `CLUSTERING_LINKAGE`: Linkage method (default: `average`)

### Duplicate Extraction Configuration
- `MIN_SIMILARITY_SCORE`: Minimum cosine similarity for duplicate pairs (default: 0.7)
- `TOP_K_PAIRS`: Number of top duplicate pairs to output (default: 100)

## Output Files

The pipeline generates the following outputs in the `output/` directory:

1. **`duplicate_pairs.json`**: JSON file with all duplicate pairs and similarity scores
2. **`duplicate_pairs.txt`**: Human-readable text file with duplicate pairs
3. **`cluster_assignments.json`**: Mapping from complaint ID to cluster ID
4. **`evaluation_report.txt`**: Comprehensive evaluation metrics
5. **`clusters_for_inspection.json`**: Sampled clusters for manual inspection

### Output Format

**duplicate_pairs.json:**
```json
[
  {
    "complaint_id_1": "1",
    "complaint_id_2": "5",
    "similarity_score": 0.89,
    "cluster_id": 3,
    "text_1": "Complaint text...",
    "text_2": "Another complaint text..."
  }
]
```

## Design Decisions

### Model Selection

**Embedding Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- Supports 50+ languages including all target languages
- Lightweight (80MB) suitable for CPU execution
- Produces 384-dimensional embeddings
- Pretrained on multilingual paraphrase data

### Clustering Algorithm

**Agglomerative Clustering with Cosine Distance**
- Deterministic and interpretable (important for academic work)
- No random initialization, ensuring reproducibility
- Scales reasonably on CPU (O(n² log n) complexity)
- Cosine distance is appropriate for normalized embeddings
- Distance threshold provides intuitive control

### Preprocessing Strategy

**Minimal Normalization**
- Normalize whitespace only
- Preserve case (important for proper nouns and multilingual text)
- Preserve punctuation (provides context and structure)
- No language-specific rules (multilingual model handles variation)

## Performance Considerations

- **Embedding Caching**: Embeddings are cached to disk to avoid recomputation
- **Incremental Processing**: New complaints are processed incrementally using cache
- **CPU Optimization**: All algorithms are optimized for CPU-only execution
- **Memory Efficiency**: Uses efficient NumPy operations and sparse representations where possible

## Evaluation

The system provides weak evaluation suitable for low-resource scenarios:

1. **Cluster Coherence**: Mean pairwise similarity within clusters
2. **Cluster Statistics**: Size distribution, singleton count, etc.
3. **Similarity Statistics**: Distribution of duplicate pair similarities
4. **Manual Inspection**: Sampled clusters exported for qualitative analysis

## Project Structure

```
.
├── main.py                 # Pipeline orchestrator
├── config.py              # Configuration management
├── data_ingestion.py      # Data loading module
├── preprocessing.py       # Text normalization
├── embedding.py           # Sentence embedding with caching
├── clustering.py          # Clustering algorithm
├── duplicate_extraction.py # Duplicate pair extraction
├── evaluation.py          # Evaluation metrics
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                 # Input data directory (create as needed)
├── output/               # Output files directory (auto-created)
└── cache/                # Embedding cache directory (auto-created)
```

## Limitations

- **No Fine-tuning**: Uses pretrained models only (by design)
- **CPU-Only**: Optimized for CPU execution, not GPU
- **Scalability**: Clustering complexity is O(n² log n), suitable for moderate dataset sizes
- **Language Support**: Limited to languages supported by the embedding model

## Citation

If you use this system in your research, please cite:

```
[Your citation information here]
```

## License

See LICENSE file for details.

## Contributing

This is a research system designed for academic submission. For questions or issues, please open an issue in the repository.

## Acknowledgments

- Sentence Transformers library for multilingual embeddings
- scikit-learn for clustering algorithms
- The multilingual NLP research community
