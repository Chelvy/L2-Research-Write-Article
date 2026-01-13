#!/usr/bin/env python3
"""
Fix duplicate 'id' fields in cell metadata caused by merge conflicts.
Keeps the second (newer) id value and removes duplicates.
"""

import json
import re

# Read the notebook as text (since JSON is invalid)
with open('integration_paradox_demo.ipynb', 'r') as f:
    content = f.read()

# Pattern to match duplicate id fields in metadata:
# "id": "VALUE1"
# "id": "VALUE2"
# We want to keep only the second one and add proper comma after it

# Replace pattern: two consecutive id fields
pattern = r'"id":\s*"([^"]+)"\s*\n\s*"id":\s*"([^"]+)"'
replacement = r'"id": "\2"'

# Fix all duplicate id occurrences
fixed_content = re.sub(pattern, replacement, content)

# Write the fixed content
with open('integration_paradox_demo.ipynb', 'w') as f:
    f.write(fixed_content)

print("✓ Fixed duplicate metadata id fields")

# Validate JSON
try:
    with open('integration_paradox_demo.ipynb', 'r') as f:
        json.load(f)
    print("✓ Notebook JSON is now valid")
except json.JSONDecodeError as e:
    print(f"✗ JSON still invalid: {e}")
    exit(1)
