#!/usr/bin/env python3
"""
Consolidate duplicate Section 10.1 cells.
Remove cells 31 (duplicate markdown header) and 32 (shorter duplicate analysis).
Keep cells 27-28 which have the comprehensive analysis.
"""

import json

with open('integration_paradox_demo.ipynb', 'r') as f:
    nb = json.load(f)

print("="*80)
print("CONSOLIDATING DUPLICATE SECTION 10.1 CELLS")
print("="*80)
print()

# Show what we're removing
print("Removing duplicate cells:")
print(f"  Cell 31 (markdown): Duplicate '10.1 Detailed Error Propagation Analysis' header")
print(f"  Cell 32 (code): Shorter duplicate error propagation analysis (48 lines)")
print()

print("Keeping:")
print(f"  Cell 27 (markdown): Original '10.1 Detailed Error Propagation Analysis' header")
print(f"  Cell 28 (code): Comprehensive error propagation analysis (138 lines)")
print()

# Remove cells 31 and 32 (remove in reverse order to preserve indices)
del nb['cells'][32]
del nb['cells'][31]

# Save
with open('integration_paradox_demo.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("✓ Removed duplicate cells 31 and 32")
print()

# Validate
with open('integration_paradox_demo.ipynb', 'r') as f:
    test_nb = json.load(f)

print(f"✓ Notebook now has {len(test_nb['cells'])} cells (was {len(nb['cells']) + 2})")

# Verify Section 10.1 is now unique
section_10_1_count = 0
for i, cell in enumerate(test_nb['cells']):
    source = ''.join(cell.get('source', []))
    if '10.1' in source and 'Detailed Error Propagation' in source and cell['cell_type'] == 'markdown':
        section_10_1_count += 1
        print(f"  Section 10.1 header found at cell {i}")

print(f"\n✓ Section 10.1 appears {section_10_1_count} time(s) (should be 1)")

if section_10_1_count == 1:
    print("\n✅ Successfully consolidated Section 10.1!")
else:
    print(f"\n⚠️  Warning: Section 10.1 appears {section_10_1_count} times")

print("="*80)
