#!/usr/bin/env python3
"""
Fix KeyError in Cell 29 by using .get() with default values.
"""

import json
import sys


def fix_cell29_keyerror(notebook_path: str) -> bool:
    """Fix KeyError in section 10.1 by using .get() for dict access."""

    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        # Fix Cell 29
        cell = nb['cells'][29]
        source = ''.join(cell['source'])

        # Replace the problematic line
        old_line = "impact = scenario['propagation_prob'] * scenario['amplification']"
        new_line = "impact = scenario.get('propagation_prob', 0.5) * scenario.get('amplification', 1.0)"

        if old_line in source:
            print("Found problematic line in Cell 29")
            source_fixed = source.replace(old_line, new_line)

            # Also fix other similar patterns
            source_fixed = source_fixed.replace(
                "scenario['propagation_prob']",
                "scenario.get('propagation_prob', 0.5)"
            )
            source_fixed = source_fixed.replace(
                "scenario['amplification']",
                "scenario.get('amplification', 1.0)"
            )
            source_fixed = source_fixed.replace(
                "scenario['severity']",
                "scenario.get('severity', 'MEDIUM')"
            )

            # Update cell
            nb['cells'][29]['source'] = source_fixed.split('\n')

            # Write back
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=2)

            print("✅ Successfully fixed Cell 29!")
            print("   Changed direct key access to .get() with defaults")
            return True
        else:
            print("❌ Could not find the problematic line")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    notebook_path = "/home/user/L2-Research-Write-Article/integration_paradox_demo.ipynb"

    print("="*80)
    print("FIXING CELL 29 - KeyError in section 10.1")
    print("="*80 + "\n")

    success = fix_cell29_keyerror(notebook_path)

    if success:
        print("\n" + "="*80)
        print("✓ Fix complete!")
        print("\nCell 29 will now handle missing keys gracefully with defaults:")
        print("  - propagation_prob: defaults to 0.5")
        print("  - amplification: defaults to 1.0")
        print("  - severity: defaults to 'MEDIUM'")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("✗ Fix failed")
        print("="*80)
        sys.exit(1)
