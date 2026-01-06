#!/usr/bin/env python3
"""
Fix section 10.3 to use the correct IntegrationMetrics methods
"""

import json

# Load the notebook
with open('integration_paradox_demo.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the cell with "10.3 Recommendations & Mitigation Strategies"
target_cell_idx = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'isolated_avg = sum(metrics.isolated_accuracy.values())' in source:
            target_cell_idx = i
            break

if target_cell_idx is None:
    print("❌ Could not find section 10.3 cell!")
    exit(1)

# Get the current source
source_lines = notebook['cells'][target_cell_idx]['source']
source_text = ''.join(source_lines)

# Replace the incorrect attribute access with method calls
old_code = """# Calculate integration gap
isolated_avg = sum(metrics.isolated_accuracy.values()) / len(metrics.isolated_accuracy)
integrated_avg = sum(metrics.integrated_accuracy.values()) / len(metrics.integrated_accuracy)
integration_gap = ((isolated_avg - integrated_avg) / isolated_avg) * 100"""

new_code = """# Calculate integration gap using IntegrationMetrics methods
isolated_accuracy = metrics.calculate_isolated_accuracy()
system_accuracy = metrics.calculate_system_accuracy()

# Calculate averages
isolated_avg = sum(isolated_accuracy.values()) / len(isolated_accuracy) if isolated_accuracy else 0
integration_gap = ((isolated_avg - system_accuracy) / isolated_avg * 100) if isolated_avg > 0 else 0"""

source_text = source_text.replace(old_code, new_code)

# Update the cell
notebook['cells'][target_cell_idx]['source'] = source_text.split('\n')

# Save the notebook
with open('integration_paradox_demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("="*80)
print("FIXING SECTION 10.3 - USE CORRECT METHODS")
print("="*80)
print()
print("✅ Successfully fixed section 10.3!")
print()
print("Changes:")
print("  ❌ metrics.isolated_accuracy (doesn't exist)")
print("  ✅ metrics.calculate_isolated_accuracy() (correct method)")
print()
print("  ❌ metrics.integrated_accuracy (doesn't exist)")
print("  ✅ metrics.calculate_system_accuracy() (correct method)")
print()
print("="*80)
print("✓ Fix complete!")
print()
print("The code now correctly calls the IntegrationMetrics methods")
print("to calculate isolated accuracy and system accuracy.")
print("="*80)
