#!/usr/bin/env python3
"""
Revert Cell 29 to use actual values instead of defaults.
"""

import json
import sys


def revert_cell29_to_actual_values(notebook_path: str) -> bool:
    """Change Cell 29 back to use actual scenario values."""

    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        # Fix Cell 29 - revert to using actual values
        cell = nb['cells'][29]
        source = ''.join(cell['source'])

        # Replace .get() with direct access since all scenarios now have these keys
        replacements = [
            ("scenario.get('propagation_prob', 0.5)", "scenario['propagation_prob']"),
            ("scenario.get('amplification', 1.0)", "scenario['amplification']"),
            ("scenario.get('severity', 'MEDIUM')", "scenario['severity']"),
            ("scenario.get('error_type', 'UNKNOWN')", "scenario['error_type']"),
        ]

        source_fixed = source
        for old, new in replacements:
            source_fixed = source_fixed.replace(old, new)

        # Update cell
        nb['cells'][29]['source'] = source_fixed.split('\n')

        # Write back
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)

        print("✅ Successfully reverted Cell 29!")
        print("   Now using actual values from complete error scenarios")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    notebook_path = "/home/user/L2-Research-Write-Article/integration_paradox_demo.ipynb"

    print("="*80)
    print("REVERTING CELL 29 - USE ACTUAL VALUES")
    print("="*80 + "\n")

    success = revert_cell29_to_actual_values(notebook_path)

    if success:
        print("\n" + "="*80)
        print("✓ Revert complete!")
        print("\nCell 29 now calculates impact using:")
        print("  • Actual propagation_prob from each scenario")
        print("  • Actual amplification from each scenario")
        print("  • Real severity levels")
        print("\nTOP 10 will show truly highest-impact errors!")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("✗ Revert failed")
        print("="*80)
        sys.exit(1)
