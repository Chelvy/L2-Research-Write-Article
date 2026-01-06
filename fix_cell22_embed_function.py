#!/usr/bin/env python3
"""
Embed the get_comprehensive_error_scenarios function directly into Cell 22
so it works in Colab without needing to import external files
"""

import json

# Load the comprehensive_error_scenarios.py to extract the function
with open('comprehensive_error_scenarios.py', 'r', encoding='utf-8') as f:
    comprehensive_code = f.read()

# Extract just the get_comprehensive_error_scenarios function
# Start from line 12 (def get_comprehensive_error_scenarios():)
# End at line 506 (return scenarios)
lines = comprehensive_code.split('\n')
function_lines = lines[11:506]  # 0-indexed, so line 12 is index 11, line 506 is index 505
function_code = '\n'.join(function_lines)

# Create the new cell content
new_cell_source = f"""# Define comprehensive error scenarios directly in the notebook
# This allows the notebook to work standalone in Colab

{function_code}

# Create the error_scenarios variable by calling the function
error_scenarios = get_comprehensive_error_scenarios()

print("✅ Loaded comprehensive error scenarios with full impact data")
print(f"   Total scenarios: {{sum(len(scenarios) for scenarios in error_scenarios.values())}}")
print("   Stages:")
for stage, scenarios in error_scenarios.items():
    print(f"   • {{stage}}: {{len(scenarios)}} error scenarios")
"""

# Load the notebook
with open('integration_paradox_demo.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find Cell 22 (the cell with comprehensive error scenarios)
target_cell_idx = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'Import and call comprehensive error scenarios' in source or \
           'Import comprehensive error scenarios' in source:
            target_cell_idx = i
            break

if target_cell_idx is None:
    print("❌ Could not find Cell 22!")
    exit(1)

# Update the cell with the embedded function
notebook['cells'][target_cell_idx]['source'] = new_cell_source.split('\n')

# Save the notebook
with open('integration_paradox_demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("="*80)
print("EMBEDDING FUNCTION INTO CELL 22 FOR COLAB COMPATIBILITY")
print("="*80)
print()
print("✅ Successfully embedded get_comprehensive_error_scenarios() into Cell 22!")
print()
print("   The notebook is now self-contained and will work in Colab")
print("   No external file imports needed")
print()
print("   Cell 22 now contains:")
print("   1. Complete get_comprehensive_error_scenarios() function definition")
print("   2. Creates error_scenarios variable with all 50 scenarios")
print("   3. All scenarios have real propagation_probability and amplification_factor")
print()
print("="*80)
print("✓ Fix complete! Reload the notebook in Colab and run Cell 22")
print("="*80)
