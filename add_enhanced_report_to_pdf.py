#!/usr/bin/env python3
"""
Add Enhanced Integration Paradox Report to PDF export.
The enhanced report (Cell 25) generates a comprehensive 9-section text report
that should be included in the PDF.
"""

import json

with open('integration_paradox_demo.ipynb', 'r') as f:
    nb = json.load(f)

cell_idx = 42
cell = nb['cells'][cell_idx]
source = ''.join(cell['source'])

# Find where page 13 ends (after recommendations)
# We'll add the enhanced report as pages 14-16

# Find the line after page 13's pdf.savefig
lines = source.split('\n')
page13_end = None
for i, line in enumerate(lines):
    if 'PAGE 13:' in line or 'Page 13:' in line:
        # Find pdf.savefig after this
        for j in range(i, min(i+60, len(lines))):
            if 'pdf.savefig' in lines[j] and 'plt.close' in lines[j+1]:
                page13_end = j + 2  # After plt.close
                break
        break

if not page13_end:
    print("❌ Could not find PAGE 13 end")
    exit(1)

print(f"Found PAGE 13 ends at line {page13_end}")

# Insert enhanced report generation code
enhanced_report_code = '''
    # ========================================================================
    # GENERATE ENHANCED INTEGRATION REPORT
    # ========================================================================
    # Create enhanced metrics instance
    enhanced = EnhancedIntegrationMetrics(metrics)

    # Generate the comprehensive report text
    enhanced_report_text = enhanced.generate_comprehensive_report(error_scenarios)

    # ========================================================================
    # PAGE 14-16: ENHANCED INTEGRATION PARADOX REPORT
    # ========================================================================
    # Split the enhanced report into pages (it's quite long)
    report_lines = enhanced_report_text.split('\\n')

    # Calculate how many lines fit per page (with margins)
    lines_per_page = 55  # Approximate for 9pt font with 1.2 line spacing

    # Split into pages
    page_num = 14
    for page_start in range(0, len(report_lines), lines_per_page):
        fig = plt.figure(figsize=(LETTER_WIDTH, LETTER_HEIGHT))
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Title only on first page
        if page_start == 0:
            fig.text(0.5, 0.95, 'Enhanced Integration Paradox Report',
                     ha='center', va='top',
                     fontsize=TITLE_SIZE, fontweight='bold',
                     fontname=TITLE_FONT, color=TITLE_COLOR)
            content_start_y = 0.92
        else:
            fig.text(0.5, 0.95, f'Enhanced Report (continued) - Page {page_num}',
                     ha='center', va='top',
                     fontsize=SUBTITLE_SIZE, fontweight='bold',
                     fontname=TITLE_FONT, color=TITLE_COLOR)
            content_start_y = 0.92

        # Get lines for this page
        page_lines = report_lines[page_start:page_start + lines_per_page]
        page_text = '\\n'.join(page_lines)

        fig.text(MARGIN_NORMALIZED, content_start_y, page_text,
                 ha='left', va='top',
                 fontsize=SMALL_SIZE, fontname=MONO_FONT,
                 linespacing=LINE_SPACING, color=TEXT_COLOR)

        plt.subplots_adjust(left=MARGIN_NORMALIZED, right=1-MARGIN_NORMALIZED,
                           top=1-MARGIN_NORMALIZED, bottom=MARGIN_NORMALIZED)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        page_num += 1

        # Limit to 3 pages max for the enhanced report
        if page_num > 16:
            break
'''

# Insert the enhanced report code before the final close
lines.insert(page13_end, enhanced_report_code)

# Update the page count in the print statement
new_source = '\n'.join(lines)
new_source = new_source.replace('print(f"   Total Pages: 13")', 'print(f"   Total Pages: 14-16 (depending on report length)")')

# Convert to list
new_source_list = [line + '\n' for line in new_source.split('\n')]

# Update cell
nb['cells'][cell_idx]['source'] = new_source_list

# Save
with open('integration_paradox_demo.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("="*80)
print("ADDED ENHANCED INTEGRATION PARADOX REPORT TO PDF")
print("="*80)
print()
print("✅ Integrated EnhancedIntegrationMetrics report generation")
print("✅ Added pages 14-16: Enhanced Integration Paradox Report")
print()
print("The PDF now includes:")
print("  • Pages 1-13: Professional formatted analysis")
print("  • Pages 14-16: Enhanced Integration Paradox Report")
print("    - 9 comprehensive sections")
print("    - Component-level accuracy")
print("    - System-level performance")
print("    - Integration paradox gap")
print("    - Error propagation analysis")
print("    - Error severity distribution")
print("    - Stage risk assessment")
print("    - Compositional failure modes")
print("    - Recommendations")
print("    - Research alignment")
print()

# Validate
try:
    with open('integration_paradox_demo.ipynb', 'r') as f:
        test_nb = json.load(f)
    print("✓ Notebook JSON is valid")

    source = ''.join(test_nb['cells'][cell_idx]['source'])
    compile(source, '<cell>', 'exec')
    print("✓ Cell compiles successfully")

except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

print("="*80)
