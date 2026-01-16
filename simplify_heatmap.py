#!/usr/bin/env python3
"""
Simplify Error Propagation Heatmap - remove complex formatting.
Use basic, robust rendering with minimal styling.
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

# Simple, clean heatmap code with minimal formatting
simple_heatmap = '''    # PAGE 5: ERROR PROPAGATION HEATMAP (LANDSCAPE)
    # ========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(LETTER_HEIGHT, LETTER_WIDTH))

    fig.suptitle('Error Propagation Heatmap Across SDLC Stages',
                fontsize=TITLE_SIZE, fontweight='bold', y=0.95)

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

    # Simple heatmap
    im = ax.imshow(propagation_matrix, cmap='YlOrRd', aspect='equal')

    # Basic ticks and labels
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(stages, rotation=45, ha='right')
    ax.set_yticklabels(stages)

    # Simple labels
    ax.set_xlabel('Target Stage', fontsize=12)
    ax.set_ylabel('Source Stage', fontsize=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Error Count', fontsize=10)

    # Add values to cells
    for i in range(5):
        for j in range(5):
            val = int(propagation_matrix[i, j])
            if val > 0:
                text_color = 'white' if val > propagation_matrix.max()/2 else 'black'
                ax.text(j, i, str(val), ha='center', va='center',
                       color=text_color, fontsize=12, fontweight='bold')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)'''

# Replace the old PAGE 5 code
del lines[page5_start:page5_end+1]
lines.insert(page5_start, simple_heatmap)

# Reconstruct source
new_source = '\n'.join(lines)
new_source_list = [line + '\n' for line in new_source.split('\n')]

# Update cell
nb['cells'][cell_idx]['source'] = new_source_list

# Save
with open('integration_paradox_demo.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("="*80)
print("SIMPLIFIED ERROR PROPAGATION HEATMAP")
print("="*80)
print()
print("✅ Removed all complex formatting")
print("✅ Using basic matplotlib defaults")
print("✅ Simple clean rendering:")
print("   • Basic imshow() with YlOrRd colormap")
print("   • Standard tick labels (no custom fonts)")
print("   • Simple 45° rotation for x-axis")
print("   • Equal aspect ratio")
print("   • Basic colorbar")
print("   • Plain value labels in cells")
print("   • Standard tight_layout()")
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
    print("✓ Simplified heatmap should render reliably")

except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

print("="*80)
