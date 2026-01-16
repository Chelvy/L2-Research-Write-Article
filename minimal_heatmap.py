#!/usr/bin/env python3
"""
Minimal Error Propagation Heatmap - absolutely no formatting.
Just plot the raw data as matplotlib would naturally render it.
"""

import json

with open('integration_paradox_demo.ipynb', 'r') as f:
    nb = json.load(f)

cell_idx = 42
cell = nb['cells'][cell_idx]
source = ''.join(cell['source'])
lines = source.split('\n')

# Find PAGE 5
page5_start = None
page5_end = None
for i, line in enumerate(lines):
    if 'PAGE 5: ERROR PROPAGATION HEATMAP' in line:
        page5_start = i
        for j in range(i, min(i+100, len(lines))):
            if 'plt.close(fig)' in lines[j]:
                page5_end = j
                break
        break

if not page5_start or not page5_end:
    print("❌ Could not find PAGE 5")
    exit(1)

print(f"Found PAGE 5: lines {page5_start} to {page5_end}")

# Minimal heatmap - just plot the data
minimal_heatmap = '''    # PAGE 5: ERROR PROPAGATION HEATMAP (LANDSCAPE)
    # ========================================================================
    fig = plt.figure(figsize=(11, 8.5))

    # Create propagation matrix
    stages = ['Requirements', 'Design', 'Implementation', 'Testing', 'Deployment']
    propagation_matrix = np.zeros((5, 5))

    if metrics.error_propagation:
        for error in metrics.error_propagation:
            source = error.get('source', '')
            target = error.get('target', '')
            try:
                source_idx = [s.lower() for s in stages].index(source.lower())
                target_idx = [s.lower() for s in stages].index(target.lower())
                propagation_matrix[source_idx][target_idx] += 1
            except (ValueError, AttributeError):
                pass

    # Plot heatmap
    plt.imshow(propagation_matrix, cmap='YlOrRd')
    plt.xticks(range(5), stages, rotation=45)
    plt.yticks(range(5), stages)
    plt.xlabel('Target Stage')
    plt.ylabel('Source Stage')
    plt.title('Error Propagation Heatmap Across SDLC Stages')
    plt.colorbar(label='Error Count')

    # Add values
    for i in range(5):
        for j in range(5):
            val = int(propagation_matrix[i, j])
            if val > 0:
                plt.text(j, i, str(val), ha='center', va='center', color='white')

    pdf.savefig(fig)
    plt.close(fig)'''

# Replace the old PAGE 5 code
del lines[page5_start:page5_end+1]
lines.insert(page5_start, minimal_heatmap)

# Reconstruct source
new_source = '\n'.join(lines)
new_source_list = [line + '\n' for line in new_source.split('\n')]

# Update cell
nb['cells'][cell_idx]['source'] = new_source_list

# Save
with open('integration_paradox_demo.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("="*80)
print("MINIMAL HEATMAP - NO FORMATTING")
print("="*80)
print()
print("✅ Removed ALL formatting")
print("✅ Using direct matplotlib pyplot calls")
print("✅ No subplots, no axes, no customization")
print("✅ Just: plt.imshow(), plt.xticks(), plt.yticks(), plt.colorbar()")
print()

# Validate
try:
    with open('integration_paradox_demo.ipynb', 'r') as f:
        test_nb = json.load(f)
    print("✓ Notebook JSON is valid")

    source = ''.join(test_nb['cells'][cell_idx]['source'])
    compile(source, '<cell>', 'exec')
    print("✓ Cell compiles successfully")
    print()
    print("✓ Heatmap will use matplotlib defaults only")

except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

print("="*80)
