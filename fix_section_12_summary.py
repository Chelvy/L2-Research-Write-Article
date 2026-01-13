#!/usr/bin/env python3
"""
Fix Section 12 - Add summary key to export_data or fix the print statement
"""

import json

with open('integration_paradox_demo.ipynb', 'r') as f:
    nb = json.load(f)

# Find Section 12 cell (cell 44)
cell = nb['cells'][44]
source = ''.join(cell['source'])

# Replace the problematic line 452
# Change from: print(json.dumps(export_data['summary'], indent=2))
# To: print a summary of the actual data

old_code = """print("\\n📊 FINAL SUMMARY:")
print(json.dumps(export_data['summary'], indent=2))"""

new_code = """print("\\n📊 FINAL SUMMARY:")
print("-"*70)
print(f"Timestamp: {export_data['timestamp']}")
print(f"Experiment: {export_data['experiment']}")
print()
print("Metrics:")
for key, value in export_data['metrics'].items():
    if isinstance(value, dict):
        print(f"  {key}:")
        for k, v in value.items():
            print(f"    {k}: {v:.1%}" if isinstance(v, float) else f"    {k}: {v}")
    else:
        print(f"  {key}: {value:.1%}" if isinstance(value, float) else f"  {key}: {value}")
print()
print(f"Total Agent Results: {len(export_data['agent_results'])}")
print(f"Total Error Propagations: {len(export_data['error_propagation'])}")
print()
print("Error Scenarios by Stage:")
for stage, info in export_data['error_scenarios_summary'].items():
    print(f"  {stage.title()}: {info['count']} errors")
    crit = info['severities'].get('CRITICAL', 0)
    high = info['severities'].get('HIGH', 0)
    if crit > 0 or high > 0:
        print(f"    (CRITICAL: {crit}, HIGH: {high})")
print("-"*70)"""

# Replace in source
new_source = source.replace(old_code, new_code)

# Convert to proper list format
new_source_list = [line + '\n' for line in new_source.split('\n')]

# Update cell
nb['cells'][44]['source'] = new_source_list

# Save
with open('integration_paradox_demo.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("✓ Fixed Section 12 summary print statement")

# Validate
try:
    with open('integration_paradox_demo.ipynb', 'r') as f:
        test_nb = json.load(f)
    source = ''.join(test_nb['cells'][44]['source'])
    compile(source, '<cell44>', 'exec')
    print("✓ Cell 44 compiles successfully")
    print("✓ Notebook JSON is valid")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)
