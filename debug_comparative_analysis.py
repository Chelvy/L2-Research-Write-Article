#!/usr/bin/env python3
"""
Find and fix the KeyError in comparative analysis section
"""

import json

# Load the notebook
with open('integration_paradox_demo.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the comparative analysis cell
target_cell_idx = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'POC 1 vs POC 2 vs POC 3: COMPARATIVE ANALYSIS' in source and 'execution_time' in source:
            target_cell_idx = i
            print(f"Found cell at index {i}")

            # Find where poc2_metrics and poc3_metrics are defined
            if 'poc2_metrics' in source:
                # Extract just the key parts
                lines = source.split('\n')
                for j, line in enumerate(lines):
                    if "'execution_time'" in line or 'execution_time' in line:
                        print(f"Line {j}: {line[:200]}")
            break

if target_cell_idx is None:
    print("❌ Could not find comparative analysis cell!")

# Now search for where poc2_metrics is defined
print("\n" + "="*80)
print("Searching for poc2_metrics definition...")
print("="*80)

for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'poc2_metrics = {' in source:
            print(f"\nFound poc2_metrics definition in cell {i}")
            # Extract the keys
            lines = source.split('\n')
            in_dict = False
            for line in lines:
                if 'poc2_metrics = {' in line or in_dict:
                    in_dict = True
                    if "'" in line or '"' in line:
                        print(line[:150])
                    if '}' in line and in_dict:
                        break
            break

print("\n" + "="*80)
print("Searching for poc3_metrics definition...")
print("="*80)

for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'poc3_metrics = {' in source:
            print(f"\nFound poc3_metrics definition in cell {i}")
            # Extract the keys
            lines = source.split('\n')
            in_dict = False
            for line in lines:
                if 'poc3_metrics = {' in line or in_dict:
                    in_dict = True
                    if "'" in line or '"' in line:
                        print(line[:150])
                    if '}' in line and in_dict:
                        break
            break
