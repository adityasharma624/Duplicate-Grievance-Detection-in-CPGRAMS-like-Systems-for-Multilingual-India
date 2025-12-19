"""
Preprocessing module.

Performs minimal text normalization to preserve semantic content.
No aggressive language-specific rules.
"""

import re
from typing import List, Dict, Any
import config


def normalize_text(text: str) -> str:
    """
    Apply minimal text normalization.
    
    Design choices:
    - Normalize whitespace: collapse multiple spaces/tabs/newlines to single space
    - Preserve case: important for proper nouns, acronyms, and multilingual text
    - Preserve punctuation: provides context and structure
    - No language-specific normalization: multilingual model handles variation
    
    Args:
        text: Raw complaint text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Normalize whitespace: collapse multiple whitespace characters to single space
    # This handles inconsistent spacing without removing structure
    if config.NORMALIZE_WHITESPACE:
        # Replace all whitespace sequences (spaces, tabs, newlines) with single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
    
    # Lowercase only if configured (default: False to preserve multilingual content)
    if config.LOWERCASE:
        text = text.lower()
    
    # Note: We do NOT remove special characters by default
    # Punctuation and special characters can be important for:
    # - Sentence boundaries
    # - Multilingual text structure
    # - Context preservation
    
    return text


def preprocess_complaints(complaints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Preprocess a list of complaints.
    
    Applies normalization to the 'text' field of each complaint.
    Preserves all other fields unchanged.
    
    Args:
        complaints: List of complaint dictionaries
        
    Returns:
        List of preprocessed complaint dictionaries
    """
    preprocessed = []
    for complaint in complaints:
        processed_complaint = complaint.copy()
        processed_complaint['text'] = normalize_text(complaint['text'])
        preprocessed.append(processed_complaint)
    
    return preprocessed

