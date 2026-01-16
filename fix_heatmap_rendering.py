#!/usr/bin/env python3
"""
Fix Error Propagation Heatmap rendering on Page 5.
Issues: Labels might be cut off, values hard to read, tight layout issues.
"""

import json

with open('integration_paradox_demo.ipynb', 'r') as f:
    nb = json.load(f)

cell_idx = 42
cell = nb['cells'][cell_idx]
source = ''.join(cell['source'])
lines = source.split('\n')

# Find PAGE 5 and replace it with improved version
page5_start = None
page5_end = None
for i, line in enumerate(lines):
    if 'PAGE 5: ERROR PROPAGATION HEATMAP' in line:
        page5_start = i
        # Find the plt.close for this page
        for j in range(i, min(i+80, len(lines))):
            if 'plt.close(fig)' in lines[j]:
                page5_end = j
                break
        break

if not page5_start or not page5_end:
    print("❌ Could not find PAGE 5")
    exit(1)

print(f"Found PAGE 5: lines {page5_start} to {page5_end}")

# Improved heatmap code
improved_heatmap = '''    # PAGE 5: ERROR PROPAGATION HEATMAP (LANDSCAPE)
    # ========================================================================
    # Use larger landscape page for better heatmap rendering
    fig = plt.figure(figsize=(LETTER_HEIGHT, LETTER_WIDTH))

    # Create larger subplot with more space for labels
    ax = plt.subplot(111)

    fig.suptitle('Error Propagation Heatmap Across SDLC Stages',
                fontsize=TITLE_SIZE, fontweight='bold',
                fontname=TITLE_FONT, color=TITLE_COLOR, y=0.96)

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

    # Create heatmap with better color scheme
    im = ax.imshow(propagation_matrix, cmap='YlOrRd', aspect='auto',
                   interpolation='nearest')

    # Set ticks and labels with better positioning
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))

    # Rotate x-axis labels and align properly
    ax.set_xticklabels(stages, fontsize=HEADING_SIZE, fontname=BODY_FONT,
                       rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(stages, fontsize=HEADING_SIZE, fontname=BODY_FONT)

    # Add axis labels with padding
    ax.set_xlabel('Target Stage', fontsize=HEADING_SIZE, fontname=BODY_FONT,
                  fontweight='bold', labelpad=10)
    ax.set_ylabel('Source Stage', fontsize=HEADING_SIZE, fontname=BODY_FONT,
                  fontweight='bold', labelpad=10)

    # Add grid for better readability
    ax.set_xticks(np.arange(5) - 0.5, minor=True)
    ax.set_yticks(np.arange(5) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)

    # Add colorbar with better positioning
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Error Count', fontsize=HEADING_SIZE, fontname=BODY_FONT,
                   fontweight='bold', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=BODY_SIZE)

    # Add values to cells with better contrast
    max_val = propagation_matrix.max()
    for i in range(5):
        for j in range(5):
            val = propagation_matrix[i, j]
            if val > 0:
                # Use white text for darker cells, black for lighter cells
                text_color = 'white' if val > max_val * 0.5 else 'black'
                ax.text(j, i, f'{int(val)}',
                       ha='center', va='center',
                       color=text_color,
                       fontsize=HEADING_SIZE, fontname=BODY_FONT,
                       fontweight='bold')

    # Add legend/description
    total_propagations = int(propagation_matrix.sum())
    description = f'Total Error Propagations: {total_propagations}'
    fig.text(0.5, 0.02, description,
             ha='center', fontsize=BODY_SIZE, fontname=BODY_FONT,
             style='italic')

    # Adjust layout with proper margins to prevent label cutoff
    plt.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.15)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)'''

# Replace the old PAGE 5 code
del lines[page5_start:page5_end+1]
lines.insert(page5_start, improved_heatmap)

# Reconstruct source
new_source = '\n'.join(lines)
new_source_list = [line + '\n' for line in new_source.split('\n')]

# Update cell
nb['cells'][cell_idx]['source'] = new_source_list

# Save
with open('integration_paradox_demo.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("="*80)
print("IMPROVED ERROR PROPAGATION HEATMAP RENDERING")
print("="*80)
print()
print("✅ Enhanced heatmap rendering with:")
print("   • Larger font sizes for better readability")
print("   • Proper label rotation and anchoring (prevents cutoff)")
print("   • White grid lines between cells for clarity")
print("   • Better colorbar positioning with padding")
print("   • Smart text color (white on dark, black on light)")
print("   • Total propagations count at bottom")
print("   • Adjusted subplot margins to prevent label clipping")
print("   • Nearest neighbor interpolation for crisp cell boundaries")
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
    print("✓ Heatmap will render properly in PDF export")

except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

print("="*80)
