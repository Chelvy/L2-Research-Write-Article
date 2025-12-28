#!/usr/bin/env python3
"""
Remove the if __name__ == "__main__" block from Cell 22 that overwrites metrics.
"""

import json
import sys


def fix_cell22(notebook_path: str) -> bool:
    """Remove the problematic if __name__ block from Cell 22."""

    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        # Find Cell 22
        cell = nb['cells'][22]
        source = ''.join(cell['source'])

        if 'if __name__ == "__main__":' not in source:
            print("✅ Cell 22 doesn't have the problematic block")
            return True

        print("Found problematic if __name__ block in Cell 22")

        # Split by lines
        lines = source.split('\n')

        # Find the if __name__ == "__main__": line
        if_main_idx = None
        for i, line in enumerate(lines):
            if 'if __name__ == "__main__":' in line:
                if_main_idx = i
                print(f"Found at line {i}")
                break

        if if_main_idx is None:
            print("Could not find if __name__ line")
            return False

        # Remove everything from that line onwards
        cleaned_lines = lines[:if_main_idx]

        # Add a note at the end
        cleaned_lines.append("")
        cleaned_lines.append("# Note: if __name__ == '__main__' block removed")
        cleaned_lines.append("# (It was overwriting the metrics object from Cell 8)")

        # Update the cell
        nb['cells'][22]['source'] = [line + '\n' for line in cleaned_lines]

        # Write back
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)

        print(f"\n✅ Successfully removed the problematic block from Cell 22!")
        print(f"   Removed {len(lines) - len(cleaned_lines)} lines")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    notebook_path = "/home/user/L2-Research-Write-Article/integration_paradox_demo.ipynb"

    print("="*80)
    print("FIXING CELL 22 - REMOVING if __name__ BLOCK")
    print("="*80 + "\n")

    success = fix_cell22(notebook_path)

    if success:
        print("\n" + "="*80)
        print("✓ Fix complete!")
        print("\nCell 22 will no longer overwrite the metrics object.")
        print("The enhanced reporting should now work correctly!")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("✗ Fix failed")
        print("="*80)
        sys.exit(1)
