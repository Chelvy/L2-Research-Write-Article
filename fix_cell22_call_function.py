#!/usr/bin/env python3
"""
Fix Cell 22 to actually CALL the function to create error_scenarios variable
"""

import json

# Load the notebook
with open('integration_paradox_demo.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find Cell 22 (the cell with "Import comprehensive error scenarios")
target_cell_idx = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'Import comprehensive error scenarios from the complete module' in source:
            target_cell_idx = i
            break

if target_cell_idx is None:
    print("❌ Could not find Cell 22!")
    exit(1)

# Replace Cell 22 with code that calls the function
new_cell_source = """# Import and call comprehensive error scenarios
from comprehensive_error_scenarios import get_comprehensive_error_scenarios

# Create the error_scenarios variable by calling the function
error_scenarios = get_comprehensive_error_scenarios()

print("✅ Loaded comprehensive error scenarios with full impact data")
print(f"   Total scenarios: {sum(len(scenarios) for scenarios in error_scenarios.values())}")
print("   Each scenario includes:")
print("   • error_type")
print("   • severity (CRITICAL, HIGH, MEDIUM, LOW)")
print("   • propagation_probability (0.0 to 1.0)")
print("   • amplification_factor (multiplier effect)")
print("   • description")
print("   • example")
print("   • cascades_to (downstream stages)")
"""

notebook['cells'][target_cell_idx]['source'] = new_cell_source.split('\n')

# Save the notebook
with open('integration_paradox_demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("="*80)
print("FIXING CELL 22 - CALLING THE FUNCTION")
print("="*80)
print()
print("✅ Successfully updated Cell 22!")
print("   Now calling get_comprehensive_error_scenarios()")
print("   Creates error_scenarios variable with full data")
print()
print("="*80)
print("✓ Fix complete!")
print()
print("Cell 22 now:")
print("  1. Imports get_comprehensive_error_scenarios()")
print("  2. CALLS the function to create error_scenarios variable")
print("  3. Provides all scenarios with complete impact data")
print("="*80)
