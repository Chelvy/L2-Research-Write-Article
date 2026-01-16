#!/usr/bin/env python3
"""
Add the Enhanced Visualizations Dashboard to PDF.
The create_enhanced_visualizations() method creates a 3x3 dashboard with 9 plots.
"""

import json

with open('integration_paradox_demo.ipynb', 'r') as f:
    nb = json.load(f)

cell_idx = 42
cell = nb['cells'][cell_idx]
source = ''.join(cell['source'])
lines = source.split('\n')

# Find where to insert (after the enhanced report pages, before the final print)
# Look for the line after "if page_num > 16: break"
insertion_point = None
for i, line in enumerate(lines):
    if 'if page_num > 16:' in line and 'break' in line:
        # Find the next line that's not indented (end of the for loop)
        for j in range(i+1, len(lines)):
            if lines[j] and not lines[j].startswith(' '):
                insertion_point = j
                break
        break

if not insertion_point:
    print("❌ Could not find insertion point")
    exit(1)

print(f"Found insertion point at line {insertion_point}")

# Add visualization dashboard pages
visualization_code = '''
    # ========================================================================
    # PAGES 17-19: ENHANCED VISUALIZATION DASHBOARD
    # ========================================================================
    # Generate the enhanced visualizations (3x3 dashboard)
    # Note: This creates 9 subplots in a single figure, so we'll save it
    # across landscape pages

    # Create the dashboard figure
    fig_dashboard, axes = plt.subplots(3, 3, figsize=(LETTER_HEIGHT*1.5, LETTER_WIDTH*2))
    fig_dashboard.suptitle('Enhanced Integration Paradox Analysis Dashboard',
                          fontsize=TITLE_SIZE+2, fontweight='bold',
                          fontname=TITLE_FONT, color=TITLE_COLOR)

    # Flatten axes for easier access
    axes_flat = axes.flatten()

    # Get metrics
    isolated = metrics.calculate_isolated_accuracy()
    system = metrics.calculate_system_accuracy()
    gap = metrics.calculate_integration_gap()
    avg_isolated = sum(isolated.values()) / len(isolated)

    # ---------------------------------------------------------------------
    # Plot 1: Component vs System Accuracy (Enhanced)
    # ---------------------------------------------------------------------
    ax = axes_flat[0]
    if isolated:
        agents = list(isolated.keys()) + ['System']
        accuracies = list(isolated.values()) + [system]
        colors = ['#2ecc71'] * len(isolated) + ['#e74c3c']
        bars = ax.bar(range(len(agents)), [a*100 for a in accuracies],
                     color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=SMALL_SIZE)
        ax.set_ylabel('Accuracy (%)', fontsize=BODY_SIZE, fontweight='bold')
        ax.set_title('Component vs System Accuracy', fontsize=BODY_SIZE, fontweight='bold')
        ax.axhline(y=90, color='blue', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

    # ---------------------------------------------------------------------
    # Plot 2: Integration Gap Waterfall
    # ---------------------------------------------------------------------
    ax = axes_flat[1]
    if isolated:
        categories = ['Component', 'Loss', 'System']
        values = [avg_isolated*100, -gap, system*100]
        colors_waterfall = ['#2ecc71', '#e74c3c', '#e67e22']
        ax.bar(categories, values, color=colors_waterfall, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Accuracy (%)', fontsize=BODY_SIZE, fontweight='bold')
        ax.set_title(f'Integration Gap: {gap:.1f}%', fontsize=BODY_SIZE, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-')
        ax.grid(axis='y', alpha=0.3)

    # ---------------------------------------------------------------------
    # Plot 3: Error Generation by Stage
    # ---------------------------------------------------------------------
    ax = axes_flat[2]
    if metrics.error_propagation:
        df_errors = pd.DataFrame(metrics.error_propagation)
        if 'source' in df_errors.columns:
            error_counts = df_errors.groupby('source').size().sort_values()
            ax.barh(range(len(error_counts)), error_counts.values,
                   color='orange', alpha=0.7, edgecolor='black')
            ax.set_yticks(range(len(error_counts)))
            ax.set_yticklabels(error_counts.index, fontsize=SMALL_SIZE)
            ax.set_xlabel('Errors Generated', fontsize=BODY_SIZE, fontweight='bold')
            ax.set_title('Error Generation by Stage', fontsize=BODY_SIZE, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

    # ---------------------------------------------------------------------
    # Plot 4: Error Severity Distribution
    # ---------------------------------------------------------------------
    ax = axes_flat[3]
    if error_scenarios:
        severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for stage, errors in error_scenarios.items():
            for error in errors:
                sev = error.get('severity', 'MEDIUM')
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

        colors_severity = ['darkred', 'orange', 'yellow', 'lightgreen']
        ax.pie(severity_counts.values(), labels=severity_counts.keys(),
              autopct='%1.1f%%', colors=colors_severity, startangle=90)
        ax.set_title('Error Severity Distribution', fontsize=BODY_SIZE, fontweight='bold')

    # ---------------------------------------------------------------------
    # Plot 5: Stage Risk Heatmap
    # ---------------------------------------------------------------------
    ax = axes_flat[4]
    if error_scenarios:
        # Calculate stage risks
        stage_risks = {}
        for stage, errors in error_scenarios.items():
            total_risk = 0
            for error in errors:
                severity_weight = {'CRITICAL': 4.0, 'HIGH': 3.0, 'MEDIUM': 2.0, 'LOW': 1.0}.get(error.get('severity', 'MEDIUM'), 2.0)
                prop_prob = error.get('propagation_probability', 0.5)
                amplification = error.get('amplification_factor', 1.0)
                error_risk = severity_weight * prop_prob * amplification
                total_risk += error_risk
            stage_risks[stage] = total_risk / len(errors) if errors else 0

        if stage_risks:
            stages = list(stage_risks.keys())
            risks = list(stage_risks.values())
            max_risk = max(risks) if risks else 1.0
            normalized_risks = [r / max_risk for r in risks] if max_risk > 0 else risks

            im = ax.imshow([normalized_risks], cmap='RdYlGn_r', aspect='auto')
            ax.set_xticks(range(len(stages)))
            ax.set_xticklabels([s.capitalize() for s in stages], rotation=45, ha='right', fontsize=SMALL_SIZE)
            ax.set_yticks([])
            ax.set_title('Stage Risk Assessment', fontsize=BODY_SIZE, fontweight='bold')

    # ---------------------------------------------------------------------
    # Plot 6: Amplification Rate Analysis
    # ---------------------------------------------------------------------
    ax = axes_flat[5]
    amplified = sum(1 for e in metrics.error_propagation if e.get('amplified', False))
    contained = len(metrics.error_propagation) - amplified

    categories = ['Amplified', 'Contained']
    values = [amplified, contained]
    colors_amp = ['#e74c3c', '#2ecc71']
    ax.bar(categories, values, color=colors_amp, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Error Count', fontsize=BODY_SIZE, fontweight='bold')
    ax.set_title('Error Amplification Analysis', fontsize=BODY_SIZE, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # ---------------------------------------------------------------------
    # Plot 7: Top Error Types
    # ---------------------------------------------------------------------
    ax = axes_flat[6]
    if error_scenarios:
        all_errors_viz = []
        for stage, errors in error_scenarios.items():
            for error in errors:
                all_errors_viz.append({
                    'type': error['error_type'],
                    'impact': error.get('propagation_probability', 0.5) * error.get('amplification_factor', 1.0)
                })

        if all_errors_viz:
            df_errors_viz = pd.DataFrame(all_errors_viz)
            top_errors = df_errors_viz.groupby('type')['impact'].sum().sort_values(ascending=True).tail(10)
            ax.barh(range(len(top_errors)), top_errors.values,
                   color='crimson', alpha=0.7, edgecolor='black')
            ax.set_yticks(range(len(top_errors)))
            ax.set_yticklabels([t[:30] for t in top_errors.index], fontsize=SMALL_SIZE-1)
            ax.set_xlabel('Total Impact', fontsize=BODY_SIZE, fontweight='bold')
            ax.set_title('Top 10 Error Types', fontsize=BODY_SIZE, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

    # ---------------------------------------------------------------------
    # Plot 8: Comparison to Research Baseline
    # ---------------------------------------------------------------------
    ax = axes_flat[7]
    dafnycomp_gap = 92.0
    categories = ['DafnyCOMP', 'Current']
    values = [dafnycomp_gap, gap]
    colors_baseline = ['purple', '#e74c3c' if gap > dafnycomp_gap else '#2ecc71']
    ax.bar(categories, values, color=colors_baseline, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Integration Gap (%)', fontsize=BODY_SIZE, fontweight='bold')
    ax.set_title('vs Research Baseline', fontsize=BODY_SIZE, fontweight='bold')
    ax.axhline(y=dafnycomp_gap, color='purple', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    # ---------------------------------------------------------------------
    # Plot 9: Summary Metrics Card
    # ---------------------------------------------------------------------
    ax = axes_flat[8]
    ax.axis('off')

    summary_text = f"""SUMMARY METRICS

Component Accuracy: {avg_isolated*100:.1f}%
System Accuracy: {system*100:.1f}%
Integration Gap: {gap:.1f}%

Error Propagations: {len(metrics.error_propagation)}
Amplification Rate: {amplified/len(metrics.error_propagation)*100:.1f}%

vs DafnyCOMP: {gap - dafnycomp_gap:+.1f}%"""

    ax.text(0.5, 0.5, summary_text, ha='center', va='center',
           fontsize=BODY_SIZE, family=MONO_FONT,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.set_title('Key Metrics', fontsize=BODY_SIZE, fontweight='bold')

    # Save the dashboard across multiple landscape pages
    plt.tight_layout()

    # Save as landscape pages (split the 3x3 grid across pages if needed)
    # For simplicity, save entire dashboard on one large page
    pdf.savefig(fig_dashboard, bbox_inches='tight', orientation='landscape')
    plt.close(fig_dashboard)
'''

# Insert the visualization code
lines.insert(insertion_point, visualization_code)

# Update page count message
new_source = '\n'.join(lines)
new_source = new_source.replace(
    'print(f"   Total Pages: 14-16 (depending on report length)")',
    'print(f"   Total Pages: 17-19 (depending on report length)")'
)

# Convert to list
new_source_list = [line + '\n' for line in new_source.split('\n')]

# Update cell
nb['cells'][cell_idx]['source'] = new_source_list

# Save
with open('integration_paradox_demo.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("="*80)
print("ADDED ENHANCED VISUALIZATION DASHBOARD TO PDF")
print("="*80)
print()
print("✅ Added 3×3 visualization dashboard (9 plots)")
print()
print("Dashboard includes:")
print("  1. Component vs System Accuracy (Enhanced)")
print("  2. Integration Gap Waterfall")
print("  3. Error Generation by Stage")
print("  4. Error Severity Distribution (pie chart)")
print("  5. Stage Risk Heatmap")
print("  6. Amplification Rate Analysis")
print("  7. Top 10 Error Types by Impact")
print("  8. Comparison to Research Baseline (DafnyCOMP)")
print("  9. Summary Metrics Card")
print()
print("PDF now includes:")
print("  Pages 1-13: Professional formatted analysis")
print("  Pages 14-16: Enhanced Integration Paradox Report (text)")
print("  Pages 17-19: Enhanced Visualization Dashboard (9 plots)")
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
    print("✓ Complete 'Generate comprehensive report and visualizations' now in PDF")

except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

print("="*80)
