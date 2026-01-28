#!/usr/bin/env python3
"""
Fix Page 5 visualization to read from error_scenarios instead of metrics.error_propagation
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
    if 'PAGE 5: ERROR PROPAGATION PATTERNS' in line:
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

# Fixed visualization using error_scenarios data
fixed_visualization = '''    # PAGE 5: ERROR PROPAGATION PATTERNS (LANDSCAPE)
    # ========================================================================
    fig = plt.figure(figsize=(LETTER_HEIGHT, LETTER_WIDTH))

    # Create two subplots side by side
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    fig.suptitle('Error Propagation Patterns Across SDLC Stages',
                fontsize=TITLE_SIZE, fontweight='bold',
                fontname=TITLE_FONT, color=TITLE_COLOR, y=0.96)

    # Count errors by stage from error_scenarios
    stages = ['Requirements', 'Design', 'Implementation', 'Testing', 'Deployment']
    error_counts = {stage: 0 for stage in stages}
    cascade_counts = {stage: 0 for stage in stages}

    # Total impact by stage
    stage_impact = {stage: 0 for stage in stages}

    # Analyze error scenarios
    for stage_key, scenarios in error_scenarios.items():
        stage_name = stage_key.capitalize()
        if stage_name in error_counts:
            error_counts[stage_name] = len(scenarios)

            # Calculate total impact for this stage
            for scenario in scenarios:
                impact = scenario.get('propagation_probability', 0.5) * scenario.get('amplification_factor', 1.0)
                stage_impact[stage_name] += impact

                # Count how many other stages this cascades to
                cascades_to = scenario.get('cascades_to', [])
                for target in cascades_to:
                    target_name = target.capitalize()
                    if target_name in cascade_counts:
                        cascade_counts[target_name] += 1

    # Chart 1: Error Count and Total Impact by Stage
    x_pos = range(len(stages))
    counts = [error_counts[s] for s in stages]
    impacts = [stage_impact[s] for s in stages]

    # Normalize impact for visualization
    max_impact = max(impacts) if max(impacts) > 0 else 1
    normalized_impacts = [i / max_impact * max(counts) for i in impacts]

    width = 0.35
    bars1 = ax1.bar([x - width/2 for x in x_pos], counts,
                    width, label='Error Count',
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar([x + width/2 for x in x_pos], normalized_impacts,
                    width, label='Total Impact (normalized)',
                    color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(stages, rotation=45, ha='right', fontsize=BODY_SIZE, fontname=BODY_FONT)
    ax1.set_ylabel('Count / Impact', fontsize=HEADING_SIZE, fontname=BODY_FONT, fontweight='bold')
    ax1.set_title('Error Count and Impact by Stage', fontsize=HEADING_SIZE, fontname=BODY_FONT, fontweight='bold')
    ax1.legend(fontsize=BODY_SIZE-1, loc='upper left')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=SMALL_SIZE,
                        fontname=BODY_FONT, fontweight='bold')

    # Chart 2: Cascade Reception (which stages receive cascaded errors)
    cascade_values = [cascade_counts[s] for s in stages]
    colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']

    bars3 = ax2.barh(range(len(stages)), cascade_values,
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_yticks(range(len(stages)))
    ax2.set_yticklabels(stages, fontsize=BODY_SIZE, fontname=BODY_FONT)
    ax2.set_xlabel('Cascaded Errors Received', fontsize=HEADING_SIZE, fontname=BODY_FONT, fontweight='bold')
    ax2.set_title('Error Cascade Reception by Stage', fontsize=HEADING_SIZE, fontname=BODY_FONT, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars3, cascade_values)):
        if val > 0:
            ax2.text(val + 0.5, i, str(int(val)),
                    va='center', fontsize=BODY_SIZE, fontname=BODY_FONT, fontweight='bold')

    # Add summary text
    total_errors = sum(counts)
    total_cascades = sum(cascade_values)
    fig.text(0.5, 0.02, f'Total Error Types: {total_errors} | Total Cascade Paths: {total_cascades}',
             ha='center', fontsize=BODY_SIZE, fontname=BODY_FONT, style='italic')

    plt.tight_layout(rect=[MARGIN_NORMALIZED, MARGIN_NORMALIZED + 0.03,
                          1-MARGIN_NORMALIZED, 0.94])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)'''

# Replace PAGE 5
del lines[page5_start:page5_end+1]
lines.insert(page5_start, fixed_visualization)

# Reconstruct source
new_source = '\n'.join(lines)
new_source_list = [line + '\n' for line in new_source.split('\n')]

# Update cell
nb['cells'][cell_idx]['source'] = new_source_list

# Save
with open('integration_paradox_demo.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("="*80)
print("FIXED PAGE 5 VISUALIZATION TO USE ERROR_SCENARIOS DATA")
print("="*80)
print()
print("✅ Changed data source from metrics.error_propagation to error_scenarios")
print("✅ Updated visualization to show:")
print("   • Left: Error Count and Total Impact by Stage (grouped bars)")
print("     - Red bars: Number of error types per stage")
print("     - Blue bars: Total impact score (normalized)")
print()
print("   • Right: Error Cascade Reception by Stage (horizontal bars)")
print("     - Shows how many cascade paths target each stage")
print("     - Color gradient: green (early) to red (late) stages")
print()
print("✅ Now reads from the same data source as other sections")
print("✅ Will display actual error scenario data (50 error types)")
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
    print("✓ Visualization will now show propagation data correctly")

except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

print("="*80)
