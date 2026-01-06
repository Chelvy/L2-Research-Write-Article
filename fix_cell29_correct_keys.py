#!/usr/bin/env python3
"""
Fix Cell 29 to use correct key names from comprehensive_error_scenarios.py
- propagation_probability (not propagation_prob)
- amplification_factor (not amplification)
"""

import json

# Load the notebook
with open('integration_paradox_demo.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find Cell 29 (10.1 Detailed Error Propagation Analysis)
target_cell_idx = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if "impact = scenario['propagation_prob']" in source:
            target_cell_idx = i
            break

if target_cell_idx is None:
    print("❌ Could not find Cell 29!")
    exit(1)

# Get the source as a list of lines
source_lines = notebook['cells'][target_cell_idx]['source']
source_text = ''.join(source_lines)

# Replace the incorrect key names with correct ones
replacements = [
    ("scenario['propagation_prob']", "scenario['propagation_probability']"),
    ("scenario['amplification']", "scenario['amplification_factor']"),
]

for old, new in replacements:
    source_text = source_text.replace(old, new)

# Update the cell
notebook['cells'][target_cell_idx]['source'] = source_text.split('\n')

# Save the notebook
with open('integration_paradox_demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("="*80)
print("FIXING CELL 29 - CORRECT KEY NAMES")
print("="*80)
print()
print("✅ Successfully updated Cell 29!")
print("   Using correct key names from comprehensive_error_scenarios.py")
print()
print("Changes:")
print("  ❌ scenario['propagation_prob']")
print("  ✅ scenario['propagation_probability']")
print()
print("  ❌ scenario['amplification']")
print("  ✅ scenario['amplification_factor']")
print()
print("="*80)
print("✓ Fix complete!")
print()
print("Cell 29 now uses the correct key names that match")
print("the data structure from comprehensive_error_scenarios.py")
print("="*80)
