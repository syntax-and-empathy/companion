import os
import re
import difflib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define file paths and stage names
stages = [
    ("Draft", "/mnt/data/01-draft-name.md"),
    ("Refined", "/mnt/data/03 - refined-name.md"),
    ("Edited", "/mnt/data/04 - edited-name.md"),
    ("Final", "/mnt/data/05 - final-name.md"),
]

# Helper to read and preprocess text
def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    text_clean = re.sub(r'```.*?```', '', text, flags=re.S)
    text_clean = re.sub(r'#.*', '', text_clean)
    return text_clean.strip()

# Gather basic metrics per stage
metrics = []
texts = {}
for name, path in stages:
    txt = load_text(path)
    texts[name] = txt
    words = re.findall(r'\b\w+\b', txt)
    sentences = re.split(r'[.!?]+', txt)
    metrics.append(
