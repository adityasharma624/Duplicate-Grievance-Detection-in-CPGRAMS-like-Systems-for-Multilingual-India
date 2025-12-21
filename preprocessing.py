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
    
    text = str(text)
    
    if config.NORMALIZE_WHITESPACE:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    if config.LOWERCASE:
        text = text.lower()
    
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

