#!/usr/bin/env python3
"""
Restore the original heatmap code from before any formatting changes.
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

# Original heatmap code from commit e31f0d3
original_heatmap = '''    # PAGE 5: ERROR PROPAGATION HEATMAP (LANDSCAPE)
    # ========================================================================
    fig = plt.figure(figsize=(LETTER_HEIGHT, LETTER_WIDTH))
    ax = plt.subplot(111)

    fig.suptitle('Error Propagation Heatmap Across SDLC Stages',
                fontsize=TITLE_SIZE, fontweight='bold',
                fontname=TITLE_FONT, color=TITLE_COLOR, y=0.98)

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

    # Create heatmap
    im = ax.imshow(propagation_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(stages, fontsize=BODY_SIZE, fontname=BODY_FONT, rotation=45, ha='right')
    ax.set_yticklabels(stages, fontsize=BODY_SIZE, fontname=BODY_FONT)
    ax.set_xlabel('Target Stage', fontsize=HEADING_SIZE, fontname=BODY_FONT, fontweight='bold')
    ax.set_ylabel('Source Stage', fontsize=HEADING_SIZE, fontname=BODY_FONT, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Error Count', fontsize=BODY_SIZE, fontname=BODY_FONT, fontweight='bold')
    cbar.ax.tick_params(labelsize=SMALL_SIZE)

    # Add values to cells
    for i in range(5):
        for j in range(5):
            if propagation_matrix[i, j] > 0:
                ax.text(j, i, f'{int(propagation_matrix[i, j])}',
                       ha='center', va='center',
                       color='white' if propagation_matrix[i, j] > propagation_matrix.max()/2 else 'black',
                       fontsize=BODY_SIZE, fontname=BODY_FONT, fontweight='bold')

    plt.tight_layout(rect=[MARGIN_NORMALIZED, MARGIN_NORMALIZED,
                          1-MARGIN_NORMALIZED, 0.96])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)'''

# Replace the current PAGE 5 code with original
del lines[page5_start:page5_end+1]
lines.insert(page5_start, original_heatmap)

# Reconstruct source
new_source = '\n'.join(lines)
new_source_list = [line + '\n' for line in new_source.split('\n')]

# Update cell
nb['cells'][cell_idx]['source'] = new_source_list

# Save
with open('integration_paradox_demo.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("="*80)
print("RESTORED ORIGINAL HEATMAP CODE")
print("="*80)
print()
print("✅ Reverted to original heatmap from commit e31f0d3")
print("✅ This is the version before any formatting changes were made")
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
    print("✓ Original heatmap code restored")

except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

print("="*80)
