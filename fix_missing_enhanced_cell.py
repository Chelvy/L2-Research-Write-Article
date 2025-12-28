#!/usr/bin/env python3
"""
Fix missing cell that creates the 'enhanced' object.
"""

import json
import sys


def fix_missing_enhanced_cell(notebook_path: str) -> bool:
    """Add the missing cell that creates the enhanced object."""

    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        # Find Cell 21 (the EnhancedIntegrationMetrics class definition)
        # We need to insert a new cell after it to create the enhanced object

        class_def_cell = None
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'class EnhancedIntegrationMetrics:' in source and 'ENHANCED INTEGRATION PARADOX REPORTING' in source:
                    class_def_cell = i
                    print(f"Found EnhancedIntegrationMetrics class definition at cell {i}")
                    break

        if class_def_cell is None:
            print("❌ Could not find EnhancedIntegrationMetrics class definition")
            return False

        # Insert a new code cell after the class definition
        # This cell will call generate_enhanced_report_and_visualizations
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate comprehensive report and visualizations\n",
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
                "print(\"  - Dynamic recommendations provided\")\n",
                "print(\"  - Research baseline comparison included\")\n",
                "print(\"=\"*80)"
            ]
        }

        # Insert the new cell right after the class definition
        insert_position = class_def_cell + 1
        nb['cells'].insert(insert_position, new_cell)

        print(f"Inserted new cell at position {insert_position}")

        # Write updated notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)

        print(f"\n✅ Successfully fixed missing enhanced cell!")
        print(f"   - New cell inserted at position {insert_position}")
        print(f"   - This cell creates the 'enhanced' object")
        print(f"   - Total notebook cells: {len(nb['cells'])}")

        return True

    except Exception as e:
        print(f"❌ Error fixing missing cell: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    notebook_path = "/home/user/L2-Research-Write-Article/integration_paradox_demo.ipynb"

    print("="*80)
    print("FIXING MISSING ENHANCED OBJECT CREATION CELL")
    print("="*80 + "\n")

    success = fix_missing_enhanced_cell(notebook_path)

    if success:
        print("\n" + "="*80)
        print("✓ Fix complete!")
        print("  Now run cells in order:")
        print("  1. Cell with EnhancedIntegrationMetrics class")
        print("  2. Cell with generate_enhanced_report_and_visualizations() call")
        print("  3. Cells 10.1, 10.2, 10.3, 10.4 will now work")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("✗ Fix failed")
        print("="*80)
        sys.exit(1)
