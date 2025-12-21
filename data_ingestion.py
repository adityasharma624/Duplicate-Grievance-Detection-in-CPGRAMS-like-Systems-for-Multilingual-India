"""
Data ingestion module.

Handles loading complaint data from CSV or JSON files.
No assumptions about data cleanliness or language.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import config


def load_complaints(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load complaints from CSV or JSON file.
    
    Args:
        file_path: Path to input file (CSV or JSON)
        
    Returns:
        List of complaint dictionaries with 'id' and 'text' fields
        
    Raises:
        ValueError: If file format is not supported or required columns are missing
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    file_ext = file_path.suffix.lower()
    
    if file_ext not in config.SUPPORTED_INPUT_FORMATS:
        raise ValueError(
            f"Unsupported file format: {file_ext}. "
            f"Supported formats: {config.SUPPORTED_INPUT_FORMATS}"
        )
    
    if file_ext == ".csv":
        df = pd.read_csv(file_path)
    elif file_ext == ".json":
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
    else:
        raise ValueError(f"Unexpected file format: {file_ext}")
    
    if config.TEXT_COLUMN_NAME not in df.columns:
        raise ValueError(
            f"Required column '{config.TEXT_COLUMN_NAME}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    
    if config.ID_COLUMN_NAME not in df.columns:
        df[config.ID_COLUMN_NAME] = range(len(df))
    
    complaints = []
    for _, row in df.iterrows():
        complaint = {
            'id': str(row[config.ID_COLUMN_NAME]),
            'text': str(row[config.TEXT_COLUMN_NAME])
        }
        for col in df.columns:
            if col not in [config.ID_COLUMN_NAME, config.TEXT_COLUMN_NAME]:
                complaint[col] = row[col]
        complaints.append(complaint)
    
    return complaints


def validate_complaints(complaints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate and filter complaints.
    
    Removes complaints with empty or invalid text.
    
    Args:
        complaints: List of complaint dictionaries
        
    Returns:
        Filtered list of valid complaints
    """
    valid_complaints = []
    for complaint in complaints:
        text = complaint.get('text', '')
        if text and str(text).strip():
            valid_complaints.append(complaint)
    
    return valid_complaints

