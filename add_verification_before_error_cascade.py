#!/usr/bin/env python3
"""
Add verification cell before Cell 20 to check metrics object status.
"""

import json
import sys


def add_verification_before_cell20(notebook_path: str) -> bool:
    """Add verification cell before comprehensive error scenarios."""

    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        # Find Cell 20 (should be the comprehensive error scenarios markdown header)
        cell20_idx = None
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'markdown':
                source = ''.join(cell['source'])
                if '## 9. Analyze Error Propagation (Enhanced)' in source:
                    cell20_idx = i
                    print(f"Found error propagation header at cell {i}")
                    break

        if cell20_idx is None:
            print("❌ Could not find Cell 20")
            return False

        # Create verification cell
        verification_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Quick check: Verify metrics object still has all methods before error cascade\n",
                "print(\"Checking metrics object before comprehensive error cascade...\")\n",
                "required = ['calculate_isolated_accuracy', 'calculate_system_accuracy', 'calculate_integration_gap']\n",
                "missing = [m for m in required if not hasattr(metrics, m)]\n",
                "if missing:\n",
                "    print(f\"❌ ERROR: metrics is missing methods: {missing}\")\n",
                "    print(\"\\n⚠️  The metrics object was overwritten!\")\n",
                "    print(\"   Re-run Cell 8 and Cell 9 to restore it.\\n\")\n",
                "    raise RuntimeError(\"metrics object was overwritten - please re-run Cell 8\")\n",
                "else:\n",
                "    print(\"✅ metrics object is intact with all required methods\\n\")\n"
            ]
        }

        # Insert right before Cell 20
        nb['cells'].insert(cell20_idx, verification_cell)
        print(f"Inserted verification cell at index {cell20_idx}")

        # Write updated notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)

        print(f"\n✅ Successfully added verification cell before error cascade!")
        print(f"   Total cells: {len(nb['cells'])}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    notebook_path = "/home/user/L2-Research-Write-Article/integration_paradox_demo.ipynb"

    print("="*80)
    print("ADDING METRICS VERIFICATION BEFORE ERROR CASCADE")
    print("="*80 + "\n")

    success = add_verification_before_cell20(notebook_path)

    if success:
        print("\n" + "="*80)
        print("✓ Verification cell added!")
        print("\nThis will help identify WHEN the metrics object gets overwritten.")
        print("Run cells in order and see which cell triggers the error.")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("✗ Failed")
        print("="*80)
        sys.exit(1)
