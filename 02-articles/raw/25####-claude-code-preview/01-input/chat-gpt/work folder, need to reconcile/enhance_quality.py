#!/usr/bin/env python3
"""
Enhanced quality pass for ai-v-human-v3.ipynb - improves existing docstrings and type hints.

Improvements:
- Enhances generic docstrings with more specific descriptions
- Fixes incorrect return types (e.g., List[Any] -> Dict[str, Any])
- Adds more detailed parameter descriptions
- Ensures all docstrings follow Google style with proper sections
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

NOTEBOOK_PATH = Path("/Users/williamtrekell/Documents/soylent-army/colab/ai-v-human-v3.ipynb")


# Specific function fixes - (cell_idx, func_name, replacement_pattern)
SPECIFIC_FIXES = [
    # Module 1 fixes
    (11, '_ensure_pkg', {
        'old_docstring': '"""Ensure required package or resource is available.        Args:        import_name: Module name.        pip_name: Function parameter.        version: Version string or number."""',
        'new_docstring': '''"""
    Ensure required package or resource is available.

    Attempts to import a module, installing it via pip if not found.

    Args:
        import_name: Name of the module to import.
        pip_name: Package name for pip (if different from import name).
        version: Version constraint string (e.g., ">=1.0.0").

    Returns:
        Imported module object.

    Raises:
        ModuleNotFoundError: If installation fails.
    """'''
    }),
    (15, 'token_zipf_stats', {
        'old_type': '-> List[Any]:',
        'new_type': '-> Tuple[Dict[str, float], List[float]]:',
        'old_docstring': '"""Calculate Zipf frequency statistics for tokens.        Args:        tokens: List of tokens."""',
        'new_docstring': '''"""
    Calculate Zipf frequency statistics for tokens.

    Computes mean, std, percentiles, and frequency distribution rates
    using wordfreq Zipf scale (0-8, where 0=rare, 8=common).

    Args:
        tokens: List of token strings to analyze.

    Returns:
        Tuple of (statistics dict, valid frequency list).
        Statistics dict contains: zipf_mean, zipf_std, zipf_p25, zipf_p50,
        zipf_p75, rare_rate, mid_rate, common_rate.
    """'''
    }),
    (15, 'latin_alpha_ratio', {
        'old_docstring': '"""Calculate ratio metric.        Args:        text: Text content to process."""',
        'new_docstring': '''"""
    Calculate ratio of Latin alphabetic characters to total text.

    Args:
        text: Input text string to analyze.

    Returns:
        Float ratio between 0.0 and 1.0.
    """'''
    }),

    # Module 2 fixes
    (20, '_ensure', {
        'old_docstring': '"""Ensure required package or resource is available.        Args:        resource: Function parameter.        locator: Function parameter."""',
        'new_docstring': '''"""
    Ensure required NLTK resource is available.

    Downloads the resource if not found locally.

    Args:
        resource: NLTK resource name (e.g., "punkt").
        locator: NLTK data path locator string.

    Raises:
        LookupError: If resource cannot be found after download.
    """'''
    }),
    (21, 'sent_spans', {
        'old_docstring': '"""Extract text spans with offsets.        Args:        doc: Text content to process."""',
        'new_docstring': '''"""
    Extract sentence spans with character offsets from spaCy doc.

    Args:
        doc: spaCy processed Doc object.

    Returns:
        List of (start_offset, end_offset) tuples for each sentence.
    """'''
    }),
    (21, 'token_spans', {
        'old_docstring': '"""Extract text spans with offsets.        Args:        doc: Text content to process."""',
        'new_docstring': '''"""
    Extract token spans with character offsets from spaCy doc.

    Args:
        doc: spaCy processed Doc object.

    Returns:
        List of (start_offset, end_offset) tuples for each token.
    """'''
    }),
    (22, 'build_windows', {
        'old_docstring': '"""Create sliding windows over data.        Args:        df: Input data.        win: Function parameter.        stride: Function parameter."""',
        'new_docstring': '''"""
    Create sliding windows over document sentences.

    Generates overlapping windows of sentences with metadata for analysis.

    Args:
        df: DataFrame with article_id, version_id, and doc columns.
        win: Window size in number of sentences.
        stride: Step size between windows.

    Returns:
        List of window dictionaries with metadata and text spans.
    """'''
    }),

    # Module 3 fixes
    (31, '_sentence_stats', {
        'old_docstring': '"""Calculate statistical features.        Args:        doc: Text content to process.        Returns:        List of processed items."""',
        'new_docstring': '''"""
    Calculate syntactic statistics for each sentence in document.

    Extracts features including token count, syntactic depth,
    dependency arc length, and entity density.

    Args:
        doc: spaCy processed Doc object.

    Returns:
        Tuple of (list of SentStats, total_token_count).
    """'''
    }),
    (35, '_basis_for_visual', {
        'old_docstring': '"""Determine basis for processing.        Args:        row: Data row to process.        Returns:        Processed result."""',
        'new_docstring': '''"""
    Determine text basis field for discourse marker visualization.

    Selects appropriate text field based on span type and availability.

    Args:
        row: DataFrame row with article metadata.

    Returns:
        String indicating basis field ("text", "text_clean", or span basis).
    """'''
    }),
    (37, '_basis_for_doc', {
        'old_docstring': '"""Determine basis for processing.        Args:        row: Data row to process."""',
        'new_docstring': '''"""
    Determine text basis field for document-level analysis.

    Prefers text_clean if available, falls back to text.

    Args:
        row: DataFrame row with article metadata.

    Returns:
        String indicating basis field ("text_clean" or "text").
    """'''
    }),

    # Module 4 fixes
    (44, 'ppl_on_sentences', {
        'old_docstring': '"""Calculate perplexity scores.        Args:        sents: Function parameter.        Returns:        Processed result."""',
        'new_docstring': '''"""
    Calculate pseudo-perplexity scores for sentences using language model.

    Uses GPT-2 to compute negative log-likelihood and perplexity metrics.

    Args:
        sents: List of sentence strings to score.

    Returns:
        DataFrame with columns: sent, nll, ppl, bpe_len.
    """'''
    }),
    (44, 'agg_stats', {
        'old_docstring': '"""Calculate statistical features.        Args:        df: Input data."""',
        'new_docstring': '''"""
    Aggregate perplexity statistics across sentences.

    Computes summary statistics (mean, std, percentiles) for PPL metrics.

    Args:
        df: DataFrame with ppl and nll columns from ppl_on_sentences.

    Returns:
        Dictionary with aggregated statistics.
    """'''
    }),

    # Module 5 fixes
    (48, 'embed', {
        'old_docstring': '"""Generate embeddings for text.        Args:        text: Text content to process."""',
        'new_docstring': '''"""
    Generate sentence embedding using SentenceTransformer model.

    Args:
        text: Input text string to embed.

    Returns:
        NumPy array of embedding vector (384 dimensions).
    """'''
    }),
    (48, 'coherence', {
        'old_docstring': '"""Calculate coherence metric.        Args:        vecs: Function parameter."""',
        'new_docstring': '''"""
    Calculate semantic coherence from cosine similarities.

    Measures how semantically similar consecutive sentences are.

    Args:
        vecs: Array of sentence embedding vectors.

    Returns:
        Float coherence score (mean pairwise cosine similarity).
    """'''
    }),
    (48, 'redundancy_from_pairwise', {
        'old_docstring': '"""Calculate redundancy metric.        Args:        pairwise_sims: Function parameter."""',
        'new_docstring': '''"""
    Calculate redundancy from pairwise similarity matrix.

    Identifies maximum similarity values indicating repeated content.

    Args:
        pairwise_sims: 2D array of pairwise cosine similarities.

    Returns:
        Dictionary with max and p90 redundancy scores.
    """'''
    }),

    # Module 6 fixes
    (55, '_make_hdb', {
        'old_docstring': '"""Helper function for make hdb.        Args:        min_cluster_size: Function parameter."""',
        'new_docstring': '''"""
    Create HDBSCAN clustering model with specified parameters.

    Args:
        min_cluster_size: Minimum number of samples in a cluster.

    Returns:
        Configured HDBSCAN model instance.
    """'''
    }),
    (55, '_topic_label_from_model', {
        'old_docstring': '"""Process topic modeling data.        Args:        topic_model: Model or tokenizer instance.        topic_id: Function parameter."""',
        'new_docstring': '''"""
    Extract human-readable label for a topic from BERTopic model.

    Args:
        topic_model: Fitted BERTopic model instance.
        topic_id: Integer topic identifier.

    Returns:
        String label for the topic (top representative words).

    Raises:
        (RuntimeError, ValueError): If topic extraction fails.
    """'''
    }),
    (56, '_topic_entropy', {
        'old_docstring': '"""Calculate entropy metric.        Args:        freqs: Function parameter."""',
        'new_docstring': '''"""
    Calculate Shannon entropy of topic distribution.

    Args:
        freqs: Array or list of topic frequency values.

    Returns:
        Float entropy value in bits.
    """'''
    }),

    # Module 7 fixes
    (63, 'window_sentences', {
        'old_docstring': '"""Create sliding windows over data.        Args:        sents: Function parameter.        win: Function parameter.        stride: Function parameter."""',
        'new_docstring': '''"""
    Create sliding windows over sentence list for paraphrase detection.

    Generates overlapping windows with configurable size and stride.

    Args:
        sents: List of (sentence_text, (start, end)) tuples.
        win: Window size in number of sentences.
        stride: Step size between windows.

    Returns:
        List of window dictionaries with sent_start_idx, sent_end_idx,
        char_span, and concatenated text.
    """'''
    }),
    (68, 'cleanup_content_dir', {
        'old_docstring': '''"""
    Removes all files and directories from the target_dir except those matching '0[1-4]-*.md'.
    Requires user confirmation before proceeding.
    """''',
        'new_docstring': '''"""
    Clean up content directory by removing non-essential files.

    Removes all files except those matching pattern '0[1-4]-*.md'.
    Requires user confirmation before proceeding.

    Args:
        target_dir: Path to directory to clean up (default: /content).

    Returns:
        None. Prints results and errors to stdout.
    """'''
    }),
]


def apply_specific_fixes(nb: Dict[str, Any]) -> int:
    """
    Apply specific targeted fixes to known functions.

    Args:
        nb: Notebook dictionary.

    Returns:
        Number of fixes applied.
    """
    fixes_applied = 0

    for cell_idx, func_name, fixes in SPECIFIC_FIXES:
        if cell_idx >= len(nb['cells']):
            continue

        cell = nb['cells'][cell_idx]
        if cell['cell_type'] != 'code':
            continue

        source = ''.join(cell['source'])
        original = source

        # Apply docstring replacement
        if 'old_docstring' in fixes and 'new_docstring' in fixes:
            source = source.replace(fixes['old_docstring'], fixes['new_docstring'])

        # Apply type hint replacement
        if 'old_type' in fixes and 'new_type' in fixes:
            # Find the function and replace its return type
            pattern = rf'(def {re.escape(func_name)}\([^)]*\))\s*{re.escape(fixes["old_type"])}'
            replacement = rf'\1 {fixes["new_type"]}'
            source = re.sub(pattern, replacement, source)

        if source != original:
            nb['cells'][cell_idx]['source'] = source.split('\n')
            fixes_applied += 1
            print(f"  Fixed cell {cell_idx}: {func_name}")

    return fixes_applied


def main():
    """Main execution function."""
    print(f"Loading notebook from {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print("\n" + "="*80)
    print("APPLYING ENHANCED QUALITY FIXES")
    print("="*80)

    fixes_applied = apply_specific_fixes(nb)

    print(f"\nSaving notebook to {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("\n" + "="*80)
    print("ENHANCEMENT SUMMARY")
    print("="*80)
    print(f"Total functions enhanced: {fixes_applied}")
    print("="*80)
    print("\nEnhanced:")
    print("  - Detailed docstrings with proper descriptions")
    print("  - Accurate return type hints")
    print("  - Complete parameter documentation")
    print("  - Proper raises sections where applicable")
    print("="*80)


if __name__ == "__main__":
    main()
