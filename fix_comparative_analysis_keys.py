#!/usr/bin/env python3
"""
Fix KeyError in comparative analysis by correcting execution_time key name
"""

import json

# Load the notebook
with open('integration_paradox_demo.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the comparative analysis cell (cell 66)
target_cell_idx = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'POC 1 vs POC 2 vs POC 3: COMPARATIVE ANALYSIS' in source and "poc2_metrics['execution_time']" in source:
            target_cell_idx = i
            break

if target_cell_idx is None:
    print("❌ Could not find comparative analysis cell!")
    exit(1)

# Get the source
source_lines = notebook['cells'][target_cell_idx]['source']
source_text = ''.join(source_lines)

# Fix the key names
# poc2_metrics has 'execution_time_seconds' not 'execution_time'
# poc3_metrics also likely has 'execution_time_seconds'
source_text = source_text.replace(
    "poc2_metrics['execution_time']",
    "poc2_metrics['execution_time_seconds']"
)

source_text = source_text.replace(
    "poc3_metrics['execution_time']",
    "poc3_metrics.get('execution_time_seconds', 0)"  # Use .get() in case poc3 is different
)

# Update the cell
notebook['cells'][target_cell_idx]['source'] = source_text.split('\n')

# Save the notebook
with open('integration_paradox_demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("="*80)
print("FIXING COMPARATIVE ANALYSIS KEY ERROR")
print("="*80)
print()
print("✅ Successfully fixed key names!")
print()
print("Changes:")
print("  ❌ poc2_metrics['execution_time']")
print("  ✅ poc2_metrics['execution_time_seconds']")
print()
print("  ❌ poc3_metrics['execution_time']")
print("  ✅ poc3_metrics.get('execution_time_seconds', 0)")
print()
print("="*80)
print("✓ Comparative analysis should now work correctly!")
print("="*80)
