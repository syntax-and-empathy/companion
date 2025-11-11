#!/usr/bin/env python3
"""Analyze remaining code quality work needed for ai-v-human-v3.ipynb."""

import json
import re
from pathlib import Path
from typing import List, Tuple

NOTEBOOK_PATH = Path("/Users/williamtrekell/Documents/soylent-army/colab/ai-v-human-v3.ipynb")


def analyze_function(source: str, func_name: str) -> dict:
    """
    Analyze a function for code quality issues.

    Args:
        source: Full cell source code.
        func_name: Name of function to analyze.

    Returns:
        Dictionary of analysis results.
    """
    # Check for docstring
    docstring_pattern = rf'^(async\s+)?def\s+{re.escape(func_name)}\s*\([^)]*\).*?:\s*\n\s+("""|\'\'\')'
    has_docstring = bool(re.search(docstring_pattern, source, re.MULTILINE | re.DOTALL))

    # Check for type hints
    type_hint_pattern = rf'^(async\s+)?def\s+{re.escape(func_name)}\s*\([^)]*\)\s*->'
    has_return_type = bool(re.search(type_hint_pattern, source, re.MULTILINE))

    # Extract function body to check for issues
    func_pattern = rf'^(async\s+)?def\s+{re.escape(func_name)}\s*\([^)]*\).*?(?=\n(?:def |class |async def |$))'
    func_match = re.search(func_pattern, source, re.MULTILINE | re.DOTALL)

    issues = []
    if not has_docstring:
        issues.append("NO_DOCSTRING")
    if not has_return_type:
        issues.append("NO_RETURN_TYPE")

    if func_match:
        func_body = func_match.group(0)

        # Check for broad exception handlers
        if re.search(r'except\s+Exception\s*:', func_body):
            issues.append("BROAD_EXCEPTION")

        # Check for long lines (>120 chars)
        long_lines = [i for i, line in enumerate(func_body.split('\n'), 1) if len(line) > 120]
        if long_lines:
            issues.append(f"LONG_LINES({len(long_lines)})")

    return {
        "name": func_name,
        "has_docstring": has_docstring,
        "has_return_type": has_return_type,
        "issues": issues
    }


def analyze_notebook():
    """Analyze the entire notebook for code quality issues."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    # Module boundaries (estimated from your description)
    modules = {
        "Module 0": (0, 8),
        "Module 1": (9, 15),
        "Module 2": (16, 30),
        "Module 3": (31, 40),
        "Module 4": (41, 50),
        "Module 5": (51, 58),
        "Module 6": (59, 65),
        "Module 7": (66, 68),
    }

    print("="*100)
    print("CODE QUALITY ANALYSIS - ai-v-human-v3.ipynb")
    print("="*100)

    total_functions = 0
    total_issues = {
        "NO_DOCSTRING": 0,
        "NO_RETURN_TYPE": 0,
        "BROAD_EXCEPTION": 0,
        "LONG_LINES": 0
    }

    for module_name, (start, end) in modules.items():
        print(f"\n{module_name} (Cells {start}-{end}):")
        print("-" * 80)

        module_funcs = []
        module_issues = []

        for cell_idx in range(start, end + 1):
            if cell_idx >= len(nb['cells']):
                continue

            cell = nb['cells'][cell_idx]
            if cell['cell_type'] != 'code':
                continue

            source = ''.join(cell['source'])

            # Find all function definitions
            func_pattern = r'^(async\s+)?def\s+(\w+)\s*\('
            functions = re.findall(func_pattern, source, re.MULTILINE)

            for async_kw, func_name in functions:
                analysis = analyze_function(source, func_name)
                if analysis['issues']:
                    module_funcs.append(f"  Cell {cell_idx}: {func_name} - {', '.join(analysis['issues'])}")
                    module_issues.extend(analysis['issues'])
                    total_functions += 1

                    # Count specific issue types
                    if "NO_DOCSTRING" in analysis['issues']:
                        total_issues["NO_DOCSTRING"] += 1
                    if "NO_RETURN_TYPE" in analysis['issues']:
                        total_issues["NO_RETURN_TYPE"] += 1
                    if "BROAD_EXCEPTION" in analysis['issues']:
                        total_issues["BROAD_EXCEPTION"] += 1
                    if any("LONG_LINES" in i for i in analysis['issues']):
                        total_issues["LONG_LINES"] += 1

        if module_funcs:
            for func_line in module_funcs:
                print(func_line)
            print(f"\n  Summary: {len(module_funcs)} functions with issues")
        else:
            print("  ✓ No issues found (or already fixed)")

    print("\n" + "="*100)
    print("OVERALL SUMMARY")
    print("="*100)
    print(f"Total functions with issues: {total_functions}")
    print(f"\nIssue breakdown:")
    print(f"  - Missing docstrings: {total_issues['NO_DOCSTRING']}")
    print(f"  - Missing return type hints: {total_issues['NO_RETURN_TYPE']}")
    print(f"  - Broad exception handlers: {total_issues['BROAD_EXCEPTION']}")
    print(f"  - Functions with long lines: {total_issues['LONG_LINES']}")
    print("="*100)

    # Estimate work required
    print("\nESTIMATED WORK REQUIRED:")
    print("-" * 80)
    doc_hours = total_issues['NO_DOCSTRING'] * 0.04  # ~2.5 min per docstring
    type_hours = total_issues['NO_RETURN_TYPE'] * 0.02  # ~1 min per type hint
    exc_hours = total_issues['BROAD_EXCEPTION'] * 0.05  # ~3 min per exception fix
    line_hours = total_issues['LONG_LINES'] * 0.02  # ~1 min per function with long lines

    print(f"  - Add docstrings: ~{doc_hours:.1f} hours ({total_issues['NO_DOCSTRING']} functions)")
    print(f"  - Add type hints: ~{type_hours:.1f} hours ({total_issues['NO_RETURN_TYPE']} functions)")
    print(f"  - Fix exceptions: ~{exc_hours:.1f} hours ({total_issues['BROAD_EXCEPTION']} handlers)")
    print(f"  - Fix long lines: ~{line_hours:.1f} hours ({total_issues['LONG_LINES']} functions)")
    print(f"\n  TOTAL ESTIMATED TIME: ~{doc_hours + type_hours + exc_hours + line_hours:.1f} hours")
    print("="*100)


if __name__ == "__main__":
    analyze_notebook()
