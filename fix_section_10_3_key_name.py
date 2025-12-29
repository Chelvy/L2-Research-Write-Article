#!/usr/bin/env python3
"""
Fix remaining propagation_prob references in section 10.3 to use propagation_probability
"""

import json

# Load the notebook
with open('integration_paradox_demo.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the cell with section 10.3 (the one that has "Stage-Specific Recommendations")
target_cell_idx = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if "propagation_prob" in source and "Stage-Specific Recommendations" in source:
            target_cell_idx = i
            break

if target_cell_idx is None:
    print("❌ Could not find the cell with propagation_prob issue!")
    exit(1)

# Get the source
source_lines = notebook['cells'][target_cell_idx]['source']
source_text = ''.join(source_lines)

# Replace propagation_prob with propagation_probability
source_text = source_text.replace("s['propagation_prob']", "s['propagation_probability']")

# Update the cell
notebook['cells'][target_cell_idx]['source'] = source_text.split('\n')

# Save the notebook
with open('integration_paradox_demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("="*80)
print("FIXING SECTION 10.3 - CORRECT KEY NAME FOR PROPAGATION")
print("="*80)
print()
print("✅ Successfully fixed section 10.3!")
print()
print("Changes:")
print("  ❌ s['propagation_prob']")
print("  ✅ s['propagation_probability']")
print()
print("="*80)
print("✓ Fix complete!")
print()
print("Section 10.3 now uses the correct key name that matches")
print("the error scenarios data structure.")
print("="*80)
