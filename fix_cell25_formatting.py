#!/usr/bin/env python3
"""
Fix Cell 25 - Enhanced Integration Reporting that got smashed into one line
"""

import json
import re

with open('integration_paradox_demo.ipynb', 'r') as f:
    nb = json.load(f)

# Get the smashed source from Cell 25
cell25_source_single = nb['cells'][25]['source'][0]

# Strategy: Add newlines before common Python patterns
# This is a heuristic approach to reconstruct the code

# First, let's add newlines before obvious patterns
source = cell25_source_single

# Add newline before comment lines (but not inline comments)
source = re.sub(r'([^\s])#', r'\1\n#', source)

# Add newline before import statements
source = re.sub(r'([a-z0-9_)])import ', r'\1\nimport ', source, flags=re.IGNORECASE)
source = re.sub(r'([a-z0-9_)])from ', r'\1\nfrom ', source, flags=re.IGNORECASE)

# Add newline before class and def
source = re.sub(r'([^\s])class ', r'\1\nclass ', source)
source = re.sub(r'([^\s])def ', r'\1\ndef ', source)

# Add newline before common indentation (4 spaces)
source = re.sub(r'([a-zA-Z0-9_"\'\)])    ([a-z])', r'\1\n    \2', source)

# Add newline before if, for, while, return
source = re.sub(r'([^\s])(if |for |while |return )', r'\1\n    \2', source)

# Add newline after colons in control structures
source = re.sub(r':    ', r':\n    ', source)

# Add newlines after closing parens/brackets that end statements
source = re.sub(r'\)    ', r')\n    ', source)
source = re.sub(r'\]    ', r']\n    ', source)

# Save intermediate result
with open('/tmp/cell25_reconstructed.py', 'w') as f:
    f.write(source)

print("Reconstructed code saved to /tmp/cell25_reconstructed.py")
print(f"Original: {len(cell25_source_single)} chars in 1 line")
print(f"Reconstructed: {len(source.split(chr(10)))} lines")

# Try to compile it
try:
    compile(source, '<cell25>', 'exec')
    print("✓ Code compiles successfully!")

    # If it compiles, update the notebook
    new_source = [line + '\n' for line in source.split('\n')]
    nb['cells'][25]['source'] = new_source

    with open('integration_paradox_demo.ipynb', 'w') as f:
        json.dump(nb, f, indent=2)

    print("✓ Updated Cell 25 in notebook")

except SyntaxError as e:
    print(f"✗ Syntax error: {e}")
    print(f"Line {e.lineno}: {e.text}")
    print("\nNeed manual reconstruction...")
    exit(1)
