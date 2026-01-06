#!/usr/bin/env python3
"""
Embed both get_comprehensive_error_scenarios and simulate_comprehensive_error_cascade
functions directly into Cell 22 for complete Colab compatibility
"""

import json

# Load the comprehensive_error_scenarios.py to extract both functions
with open('comprehensive_error_scenarios.py', 'r', encoding='utf-8') as f:
    comprehensive_code = f.read()

lines = comprehensive_code.split('\n')

# Extract get_comprehensive_error_scenarios (lines 12-506, 0-indexed: 11-505)
function1_lines = lines[11:506]
function1_code = '\n'.join(function1_lines)

# Extract simulate_comprehensive_error_cascade (lines 509-661, 0-indexed: 508-660)
function2_lines = lines[508:661]
function2_code = '\n'.join(function2_lines)

# Create the new cell content with both functions
new_cell_source = f"""# Define comprehensive error scenario functions directly in the notebook
# This allows the notebook to work standalone in Colab

{function1_code}

{function2_code}

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

# Find Cell 22
target_cell_idx = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'Define comprehensive error scenarios directly in the notebook' in source or \
           'get_comprehensive_error_scenarios' in source and 'def get_comprehensive_error_scenarios' in source:
            target_cell_idx = i
            break

if target_cell_idx is None:
    print("❌ Could not find Cell 22!")
    exit(1)

# Update the cell with both embedded functions
notebook['cells'][target_cell_idx]['source'] = new_cell_source.split('\n')

# Save the notebook
with open('integration_paradox_demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("="*80)
print("EMBEDDING BOTH FUNCTIONS INTO CELL 22 FOR COLAB COMPATIBILITY")
print("="*80)
print()
print("✅ Successfully embedded both functions into Cell 22!")
print()
print("   Functions included:")
print("   1. get_comprehensive_error_scenarios() - Returns all 50 error scenarios")
print("   2. simulate_comprehensive_error_cascade() - Simulates error propagation")
print()
print("   The notebook is now completely self-contained for Colab")
print("   No external imports needed")
print()
print("="*80)
print("✓ Fix complete! Reload notebook in Colab and run Cell 22")
print("="*80)
