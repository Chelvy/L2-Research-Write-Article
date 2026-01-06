#!/usr/bin/env python3
"""
Add verification cell immediately after Cell 8 to confirm metrics was created correctly.
"""

import json
import sys


def add_metrics_verification_after_cell8(notebook_path: str) -> bool:
    """Add a verification cell right after Cell 8."""

    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        # Find Cell 8 (IntegrationMetrics definition)
        cell8_idx = 8

        # Verify it's the right cell
        source = ''.join(nb['cells'][cell8_idx]['source'])
        if 'class IntegrationMetrics' not in source:
            print(f"❌ Cell 8 doesn't contain IntegrationMetrics class!")
            return False

        print(f"✅ Found IntegrationMetrics class at Cell {cell8_idx}")

        # Create verification cell to insert right after Cell 8
        verification_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ============================================================================\n",
                "# VERIFY: Check that metrics was created correctly\n",
                "# ============================================================================\n",
                "\n",
                "print(\"\\n\" + \"=\"*70)\n",
                "print(\"VERIFYING METRICS OBJECT FROM CELL 8\")\n",
                "print(\"=\"*70 + \"\\n\")\n",
                "\n",
                "# Check if metrics exists\n",
                "try:\n",
                "    metrics\n",
                "    print(\"✅ metrics variable exists\")\n",
                "except NameError:\n",
                "    print(\"❌ ERROR: metrics variable not found!\")\n",
                "    print(\"   Cell 8 may have failed. Please re-run Cell 8.\\n\")\n",
                "    raise\n",
                "\n",
                "# Check type\n",
                "print(f\"   Type: {type(metrics).__name__}\\n\")\n",
                "\n",
                "# Check for ALL required methods\n",
                "required_methods = [\n",
                "    'calculate_isolated_accuracy',\n",
                "    'calculate_system_accuracy',\n",
                "    'calculate_integration_gap',\n",
                "    'generate_report',\n",
                "    'visualize_results',\n",
                "    'record_agent_output',\n",
                "    'record_error_propagation'\n",
                "]\n",
                "\n",
                "print(\"Checking methods:\")\n",
                "all_present = True\n",
                "for method in required_methods:\n",
                "    has_it = hasattr(metrics, method)\n",
                "    status = \"✅\" if has_it else \"❌\"\n",
                "    print(f\"   {status} {method}\")\n",
                "    if not has_it:\n",
                "        all_present = False\n",
                "\n",
                "print(\"\\n\" + \"=\"*70)\n",
                "\n",
                "if all_present:\n",
                "    print(\"✅ SUCCESS: All methods present! Cell 8 executed correctly.\")\n",
                "    print(\"   You can now proceed to run PoC cells (9-17).\")\n",
                "else:\n",
                "    print(\"❌ ERROR: Some methods are missing!\")\n",
                "    print(\"   Cell 8 did not execute properly.\")\n",
                "    print(\"\\n💡 SOLUTION: Re-run Cell 8 and then run this cell again.\")\n",
                "    raise RuntimeError(\"metrics object was not created correctly by Cell 8\")\n",
                "\n",
                "print(\"=\"*70 + \"\\n\")\n"
            ]
        }

        # Insert right after Cell 8
        insert_idx = cell8_idx + 1
        nb['cells'].insert(insert_idx, verification_cell)

        print(f"Inserted verification cell at index {insert_idx}")
        print(f"Total cells: {len(nb['cells'])}")

        # Write updated notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)

        print(f"\n✅ Successfully added verification cell after Cell 8!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    notebook_path = "/home/user/L2-Research-Write-Article/integration_paradox_demo.ipynb"

    print("="*80)
    print("ADDING METRICS VERIFICATION CELL AFTER CELL 8")
    print("="*80 + "\n")

    success = add_metrics_verification_after_cell8(notebook_path)

    if success:
        print("\n" + "="*80)
        print("✓ Verification cell added!")
        print("\n📋 New workflow:")
        print("   1. Run Cell 8 (creates metrics)")
        print("   2. Run new Cell 9 (verifies metrics was created correctly)")
        print("   3. If Cell 9 shows ✅ SUCCESS, continue with other cells")
        print("   4. If Cell 9 shows ❌ ERROR, re-run Cell 8")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("✗ Failed")
        print("="*80)
        sys.exit(1)
