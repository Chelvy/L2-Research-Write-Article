#!/usr/bin/env python3
"""
Add verification check to enhanced reporting to ensure metrics object is valid.
"""

import json
import sys


def add_metrics_verification(notebook_path: str) -> bool:
    """Add verification to Cell 22 before calling enhanced reporting."""

    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        # Find Cell 22 (the one that calls generate_enhanced_report_and_visualizations)
        target_cell = None
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'enhanced = generate_enhanced_report_and_visualizations(metrics, error_scenarios)' in source:
                    target_cell = i
                    print(f"Found report generation call at cell {i}")
                    break

        if target_cell is None:
            print("❌ Could not find report generation cell")
            return False

        # Create new cell content with verification
        new_source = [
            "# Generate comprehensive report and visualizations\n",
            "\n",
            "# First, verify that metrics object has required methods\n",
            "required_methods = ['calculate_isolated_accuracy', 'calculate_system_accuracy', 'calculate_integration_gap']\n",
            "missing_methods = [m for m in required_methods if not hasattr(metrics, m)]\n",
            "\n",
            "if missing_methods:\n",
            "    print(\"\\n\" + \"=\"*80)\n",
            "    print(\"❌ ERROR: metrics object is missing required methods!\")\n",
            "    print(\"=\"*80)\n",
            "    print(f\"\\nMissing methods: {', '.join(missing_methods)}\\n\")\n",
            "    print(\"💡 SOLUTION:\")\n",
            "    print(\"   1. Run Cell 8 first (defines IntegrationMetrics class)\")\n",
            "    print(\"   2. Then run PoC 1 cells (9-17) to populate the metrics\")\n",
            "    print(\"   3. Then run this cell again\\n\")\n",
            "    print(\"=\"*80)\n",
            "    raise AttributeError(f\"IntegrationMetrics object missing methods: {missing_methods}\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"GENERATING ENHANCED INTEGRATION PARADOX REPORT\")\n",
            "print(\"=\"*80 + \"\\n\")\n",
            "\n",
            "enhanced = generate_enhanced_report_and_visualizations(metrics, error_scenarios)\n",
            "\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"✓ Enhanced report generation complete!\")\n",
            "print(\"  - Comprehensive 9-section report generated\")\n",
            "print(\"  - 3x3 visualization dashboard created\")\n",
            "  - Dynamic recommendations provided\")\n",
            "print(\"  - Research baseline comparison included\")\n",
            "print(\"=\"*80)"
        ]

        # Update the cell
        nb['cells'][target_cell]['source'] = new_source

        # Write updated notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)

        print(f"\n✅ Successfully added metrics verification!")
        print(f"   - Cell {target_cell} now checks for required methods")
        print(f"   - Provides helpful error message if methods are missing")

        return True

    except Exception as e:
        print(f"❌ Error adding verification: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    notebook_path = "/home/user/L2-Research-Write-Article/integration_paradox_demo.ipynb"

    print("="*80)
    print("ADDING METRICS VERIFICATION TO ENHANCED REPORTING")
    print("="*80 + "\n")

    success = add_metrics_verification(notebook_path)

    if success:
        print("\n" + "="*80)
        print("✓ Fix complete!")
        print("\n📋 Now when you run the cell:")
        print("   - It will check if metrics has the required methods")
        print("   - If not, it shows a helpful error with instructions")
        print("   - If yes, it proceeds with report generation")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("✗ Fix failed")
        print("="*80)
        sys.exit(1)
