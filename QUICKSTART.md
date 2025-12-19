# Quick Start Guide

## Step 1: Install Dependencies

Open your terminal and run:

```bash
pip install -r requirements.txt
```

This will install all the required Python packages. Wait for it to finish (it might take a few minutes the first time).

## Step 2: Prepare Your Data

You need a CSV or JSON file with your complaints. The file must have at least two columns/fields:
- `id`: A unique identifier for each complaint
- `text`: The complaint text (can be in any supported language)

### Example CSV file (`data/complaints.csv`):

```csv
id,text
1,"Water supply issue in Sector 5, not getting water for 3 days"
2,"No water available in Sector 5 area since Monday"
3,"Road condition is very poor in Main Street, many potholes"
4,"Main Street has severe potholes causing traffic problems"
```

### Example JSON file (`data/complaints.json`):

```json
[
  {"id": "1", "text": "Water supply issue in Sector 5, not getting water for 3 days"},
  {"id": "2", "text": "No water available in Sector 5 area since Monday"},
  {"id": "3", "text": "Road condition is very poor in Main Street, many potholes"},
  {"id": "4", "text": "Main Street has severe potholes causing traffic problems"}
]
```

**Important:** 
- Save your file in the `data/` folder (create it if it doesn't exist)
- Make sure your file has the `id` and `text` columns/fields

## Step 3: Run the Pipeline

### Basic usage (with CSV):

```bash
python main.py --input data/complaints.csv
```

### With JSON file:

```bash
python main.py --input data/complaints.json
```

### Save results to a custom folder:

```bash
python main.py --input data/complaints.csv --output-dir my_results
```

### Skip evaluation (faster, but no evaluation report):

```bash
python main.py --input data/complaints.csv --skip-evaluation
```

## Step 4: Check Your Results

After running, you'll find results in the `output/` folder:

- **`duplicate_pairs.json`** - All duplicate pairs with similarity scores (JSON format)
- **`duplicate_pairs.txt`** - Same information in human-readable text format
- **`cluster_assignments.json`** - Which complaints are in which cluster
- **`evaluation_report.txt`** - Statistics and metrics about the results
- **`clusters_for_inspection.json`** - Sample clusters for manual review

## Troubleshooting

### "No module named 'sentence_transformers'"
→ Run `pip install -r requirements.txt` again

### "Input file not found"
→ Make sure your file path is correct. Use `data/complaints.csv` if your file is in the `data/` folder

### "Required column 'text' not found"
→ Make sure your CSV has a column named `text` (lowercase) or your JSON has a field named `text`

## Need Help?

Run this to see all available options:
```bash
python main.py --help
```

