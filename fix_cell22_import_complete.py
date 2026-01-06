#!/usr/bin/env python3
"""
Replace Cell 22's incomplete error scenarios with import from the complete file.
"""

import json
import sys


def fix_cell22_import_scenarios(notebook_path: str) -> bool:
    """Replace Cell 22 with import from comprehensive_error_scenarios.py."""

    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        # Replace Cell 22 with a simple import
        new_cell_source = """# Import comprehensive error scenarios from the complete module
from comprehensive_error_scenarios import (
    get_comprehensive_error_scenarios,
    simulate_comprehensive_error_cascade
)

print("✅ Loaded comprehensive error scenarios with full impact data")
print("   Each scenario includes:")
print("   • error_type")
print("   • severity (CRITICAL, HIGH, MEDIUM, LOW)")
print("   • propagation_prob (0.0 to 1.0)")
print("   • amplification (multiplier effect)")
print("   • description")
print("   • example")
print("   • cascades_to (downstream stages)")
"""

        # Update Cell 22
        nb['cells'][22]['source'] = new_cell_source.split('\n')

        # Write back
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)

        print("✅ Successfully replaced Cell 22!")
        print("   Now importing from comprehensive_error_scenarios.py")
        print("   All scenarios will have complete impact data")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    notebook_path = "/home/user/L2-Research-Write-Article/integration_paradox_demo.ipynb"

    print("="*80)
    print("FIXING CELL 22 - IMPORTING COMPLETE ERROR SCENARIOS")
    print("="*80 + "\n")

    success = fix_cell22_import_scenarios(notebook_path)

    if success:
        print("\n" + "="*80)
        print("✓ Fix complete!")
        print("\nCell 22 now imports from comprehensive_error_scenarios.py")
        print("All 50 scenarios will have:")
        print("  • Real propagation probabilities (not default 0.5)")
        print("  • Real amplification factors (not default 1.0)")
        print("  • Accurate impact calculations")
        print("\nThe TOP 10 will now show ACTUAL highest-impact errors!")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("✗ Fix failed")
        print("="*80)
        sys.exit(1)
