#!/usr/bin/env python3
"""
Extract metrics definitions to understand the structure
"""

import json
import re

# Load the notebook
with open('integration_paradox_demo.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Cell 55 - poc2_metrics
print("="*80)
print("CELL 55 - POC2 METRICS")
print("="*80)
cell_55_source = ''.join(notebook['cells'][55].get('source', []))

# Find all dictionary keys in poc2_metrics
poc2_keys = re.findall(r"'([^']+)':\s*[^,\n]+", cell_55_source)
print("Keys found in poc2_metrics context:")
for key in set(poc2_keys):
    if not key.startswith('_'):
        print(f"  - {key}")

# Now find the comparative analysis cell (66)
print("\n" + "="*80)
print("CELL 66 - COMPARATIVE ANALYSIS (PROBLEM CELL)")
print("="*80)
cell_66_source = ''.join(notebook['cells'][66].get('source', []))

# Find the lines with execution_time
lines = cell_66_source.split('\n')
for i, line in enumerate(lines):
    if 'execution_time' in line:
        print(f"Line {i}: {line}")

print("\n" + "="*80)
print("Looking for actual metric variable assignments...")
print("="*80)

# Search for where poc2_metrics is actually assigned/updated
for i, cell in enumerate(notebook['cells']):
    source = ''.join(cell.get('source', []))
    if 'poc2_metrics' in source and '=' in source:
        # Look for lines with assignments
        for line in source.split('\n'):
            if 'poc2_metrics[' in line or 'poc2_metrics = ' in line:
                print(f"Cell {i}: {line[:100]}")
