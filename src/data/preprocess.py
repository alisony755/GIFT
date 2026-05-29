import re
from typing import List

# Cleans text, standardizing format
def clean_text(text: str) -> str:
    text = text.lower() # Convert to lowercase
    text = re.sub(r"http\S+", "", text) # Remove urls
    text = re.sub(r"[^a-z\s]", " ", text) # Keep letters only
    text = re.sub(r"\s+", " ", text).strip() # Remove/standardize whitespace
    return text

# Converts raw texts to token lists
def preprocess_corpus(texts: List[str]) -> List[List[str]]:
    return [clean_text(t).split() for t in texts]