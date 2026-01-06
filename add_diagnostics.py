#!/usr/bin/env python3
"""
Add detailed diagnostics to identify why metrics verification is failing.
"""

import json
import sys


def add_detailed_diagnostics(notebook_path: str) -> bool:
    """Add diagnostic cell before Cell 22 to debug metrics object."""

    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        # Find Cell 22 (the verification cell)
        target_idx = None
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'required_methods = [' in source and 'calculate_isolated_accuracy' in source:
                    target_idx = i
                    print(f"Found verification cell at index {i}")
                    break

        if target_idx is None:
            print("❌ Could not find verification cell")
            return False

        # Create diagnostic cell
        diagnostic_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ============================================================================\n",
                "# DIAGNOSTICS: Check metrics object\n",
                "# ============================================================================\n",
                "\n",
                "print(\"\\n\" + \"=\"*80)\n",
                "print(\"METRICS OBJECT DIAGNOSTICS\")\n",
                "print(\"=\"*80 + \"\\n\")\n",
                "\n",
                "# Check if metrics exists\n",
                "try:\n",
                "    metrics\n",
                "    print(\"✅ 'metrics' variable exists\")\n",
                "except NameError:\n",
                "    print(\"❌ 'metrics' variable not defined!\")\n",
                "    print(\"   Run Cell 8 first to create metrics object\\n\")\n",
                "    raise\n",
                "\n",
                "# Check type\n",
                "print(f\"   Type: {type(metrics)}\")\n",
                "print(f\"   Module: {type(metrics).__module__}\")\n",
                "print(f\"   Class: {type(metrics).__name__}\\n\")\n",
                "\n",
                "# Check available methods\n",
                "print(\"Available methods on metrics object:\")\n",
                "methods = [m for m in dir(metrics) if not m.startswith('_')]\n",
                "for method in methods:\n",
                "    print(f\"   • {method}\")\n",
                "\n",
                "print(\"\\n\" + \"-\"*80 + \"\\n\")\n",
                "\n",
                "# Check for specific required methods\n",
                "required = ['calculate_isolated_accuracy', 'calculate_system_accuracy', 'calculate_integration_gap']\n",
                "print(\"Checking for required methods:\")\n",
                "all_present = True\n",
                "for method in required:\n",
                "    has_it = hasattr(metrics, method)\n",
                "    status = \"✅\" if has_it else \"❌\"\n",
                "    print(f\"   {status} {method}: {has_it}\")\n",
                "    if not has_it:\n",
                "        all_present = False\n",
                "\n",
                "print(\"\\n\" + \"-\"*80 + \"\\n\")\n",
                "\n",
                "if all_present:\n",
                "    print(\"✅ All required methods are present!\")\n",
                "    print(\"   You can proceed to run the next cell.\\n\")\n",
                "else:\n",
                "    print(\"❌ Some methods are missing!\\n\")\n",
                "    print(\"💡 TROUBLESHOOTING:\")\n",
                "    print(\"   1. Make sure you ran Cell 8 (not just read it)\")\n",
                "    print(\"   2. Check that Cell 8 output shows: '✅ Metrics tracking framework initialized!'\")\n",
                "    print(\"   3. Try re-running Cell 8\")\n",
                "    print(\"   4. Then run this diagnostic cell again\\n\")\n",
                "    \n",
                "    # Check if it's the stub version\n",
                "    if not hasattr(metrics, 'calculate_isolated_accuracy'):\n",
                "        print(\"⚠️  This looks like a minimal/stub IntegrationMetrics object!\")\n",
                "        print(\"   The full version should be defined in Cell 8.\\n\")\n",
                "\n",
                "print(\"=\"*80)\n"
            ]
        }

        # Insert diagnostic cell RIGHT BEFORE the verification cell
        nb['cells'].insert(target_idx, diagnostic_cell)
        print(f"Inserted diagnostic cell at index {target_idx}")
        print(f"Verification cell moved to index {target_idx + 1}")

        # Write updated notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)

        print(f"\n✅ Successfully added diagnostic cell!")
        print(f"   Total cells now: {len(nb['cells'])}")

        return True

    except Exception as e:
        print(f"❌ Error adding diagnostics: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    notebook_path = "/home/user/L2-Research-Write-Article/integration_paradox_demo.ipynb"

    print("="*80)
    print("ADDING DETAILED DIAGNOSTICS CELL")
    print("="*80 + "\n")

    success = add_detailed_diagnostics(notebook_path)

    if success:
        print("\n" + "="*80)
        print("✓ Diagnostic cell added!")
        print("\n📋 Next steps:")
        print("   1. Run the new diagnostic cell (before the verification cell)")
        print("   2. It will show you:")
        print("      - Whether metrics exists")
        print("      - What type it is")
        print("      - What methods it has")
        print("      - Which required methods are missing")
        print("   3. This will help identify the problem")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("✗ Failed to add diagnostic cell")
        print("="*80)
        sys.exit(1)
