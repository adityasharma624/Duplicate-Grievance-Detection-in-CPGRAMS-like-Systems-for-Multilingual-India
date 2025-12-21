import re
from typing import List, Dict, Any
import config


def normalize_text(text: str) -> str:
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
    preprocessed = []
    for complaint in complaints:
        processed_complaint = complaint.copy()
        processed_complaint['text'] = normalize_text(complaint['text'])
        preprocessed.append(processed_complaint)
    
    return preprocessed

