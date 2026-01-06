#!/usr/bin/env python3
"""
Fix enhanced reporting to handle missing metrics methods gracefully.
"""

import json
import sys


def create_robust_enhanced_reporting():
    """Create enhanced reporting code that checks for required methods."""

    code = '''# ============================================================================
# ENHANCED INTEGRATION PARADOX REPORTING (ROBUST VERSION)
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any


class EnhancedIntegrationMetrics:
    """Enhanced metrics tracking with comprehensive analysis capabilities."""

    def __init__(self, base_metrics):
        """
        Initialize with existing IntegrationMetrics instance.

        Args:
            base_metrics: Existing IntegrationMetrics object from PoC 1
        """
        self.base_metrics = base_metrics

        # Verify required methods exist
        required_methods = [
            'calculate_isolated_accuracy',
            'calculate_system_accuracy',
            'calculate_integration_gap'
        ]

        missing_methods = []
        for method in required_methods:
            if not hasattr(base_metrics, method):
                missing_methods.append(method)

        if missing_methods:
            raise AttributeError(
                f"\\n\\n{'='*70}\\n"
                f"❌ ERROR: IntegrationMetrics object is missing required methods!\\n\\n"
                f"Missing methods: {', '.join(missing_methods)}\\n\\n"
                f"💡 SOLUTION:\\n"
                f"   Make sure you run Cell 8 first, which defines the complete\\n"
                f"   IntegrationMetrics class with all required methods.\\n\\n"
                f"   Then run the PoC cells (9-17) to populate the metrics object.\\n"
                f"{'='*70}\\n"
            )

    def analyze_error_propagation(self):
        """Analyze error propagation patterns in detail."""
        if not hasattr(self.base_metrics, 'error_propagation') or not self.base_metrics.error_propagation:
            return {
                'total_propagations': 0,
                'amplified_count': 0,
                'average_amplification_rate': 0.0,
                'amplifying_errors': 0,
                'contained_errors': 0,
                'propagation_patterns': {}
            }

        df = pd.DataFrame(self.base_metrics.error_propagation)

        amplified_count = df['amplified'].sum() if 'amplified' in df.columns else 0
        total_count = len(df)

        # Count propagation patterns
        propagation_patterns = {}
        if 'source' in df.columns and 'target' in df.columns:
            for _, row in df.iterrows():
                pattern = f"{row['source']} → {row['target']}"
                propagation_patterns[pattern] = propagation_patterns.get(pattern, 0) + 1

        analysis = {
            'total_propagations': total_count,
            'amplified_count': int(amplified_count),
            'average_amplification_rate': amplified_count / total_count if total_count > 0 else 0.0,
            'amplifying_errors': int(amplified_count),
            'contained_errors': total_count - int(amplified_count),
            'propagation_patterns': propagation_patterns
        }

        return analysis

    def calculate_error_severity_distribution(self, error_scenarios):
        """Calculate distribution of errors by severity."""
        severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}

        if error_scenarios:
            for stage, errors in error_scenarios.items():
                for error in errors:
                    severity = error.get('severity', 'MEDIUM')
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return severity_counts

    def calculate_stage_risk_scores(self, error_scenarios):
        """Calculate risk scores for each SDLC stage."""
        stage_risks = {}

        if error_scenarios:
            for stage, errors in error_scenarios.items():
                total_risk = 0
                for error in errors:
                    # Risk = severity weight × propagation probability × amplification
                    severity_weight = {
                        'CRITICAL': 4.0,
                        'HIGH': 3.0,
                        'MEDIUM': 2.0,
                        'LOW': 1.0
                    }.get(error.get('severity', 'MEDIUM'), 2.0)

                    prop_prob = error.get('propagation_prob', 0.5)
                    amplification = error.get('amplification', 1.0)

                    error_risk = severity_weight * prop_prob * amplification
                    total_risk += error_risk

                # Return normalized risk score
                stage_risks[stage] = total_risk / len(errors) if errors else 0

        return stage_risks

    def generate_comprehensive_report(self, error_scenarios=None) -> str:
        """Generate comprehensive Integration Paradox report."""
        report = []
        report.append("╔" + "═"*68 + "╗")
        report.append("║" + " "*15 + "INTEGRATION PARADOX ANALYSIS REPORT" + " "*18 + "║")
        report.append("║" + " "*20 + "Comprehensive Edition" + " "*26 + "║")
        report.append("╚" + "═"*68 + "╝")
        report.append("")

        # Timestamp
        report.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # ================================================================
        # SECTION 1: COMPONENT-LEVEL ACCURACY
        # ================================================================
        report.append("═"*70)
        report.append("📊 SECTION 1: COMPONENT-LEVEL ACCURACY (Isolated Performance)")
        report.append("═"*70)
        report.append("")

        isolated = self.base_metrics.calculate_isolated_accuracy()

        if isolated:
            report.append("Individual Agent Performance:")
            report.append("─" * 70)

            for agent, accuracy in sorted(isolated.items(), key=lambda x: x[1], reverse=True):
                bar_length = int(accuracy * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                percentage = accuracy * 100

                status = "✓" if accuracy >= 0.9 else "⚠" if accuracy >= 0.7 else "✗"
                report.append(f"  {status} {agent:30s} [{bar}] {percentage:5.1f}%")

            avg_isolated = sum(isolated.values()) / len(isolated)
            report.append("")
            report.append(f"📈 Average Isolated Accuracy: {avg_isolated*100:.1f}%")

            # Performance distribution
            excellent = sum(1 for acc in isolated.values() if acc >= 0.9)
            good = sum(1 for acc in isolated.values() if 0.7 <= acc < 0.9)
            poor = sum(1 for acc in isolated.values() if acc < 0.7)

            report.append(f"   • Excellent (≥90%): {excellent} agents")
            report.append(f"   • Good (70-89%):    {good} agents")
            report.append(f"   • Poor (<70%):      {poor} agents")
        else:
            report.append("⚠️  No isolated accuracy data available")
            report.append("")
            report.append("💡 Make sure you've run PoC 1 cells (9-17) to populate metrics!")

        report.append("")

        # [Rest of the report sections remain the same as formatted_enhanced_reporting.py]
        # ... (continuing with the same code)
'''

    # Read the rest from formatted_enhanced_reporting.py
    try:
        with open('/home/user/L2-Research-Write-Article/formatted_enhanced_reporting.py', 'r') as f:
            full_code = f.read()

        # Extract everything after the calculate_stage_risk_scores method
        # and before the final print statements
        start_marker = "def generate_comprehensive_report(self, error_scenarios=None) -> str:"
        end_marker = "def generate_enhanced_report_and_visualizations"

        start_idx = full_code.find(start_marker)
        if start_idx == -1:
            print("❌ Could not find report generation method")
            return None

        # Find where the method ends (next def at same indentation or the helper function)
        end_idx = full_code.find(end_marker, start_idx)
        if end_idx == -1:
            end_idx = len(full_code)

        # Get just the method body after our custom start
        method_code = full_code[start_idx:end_idx]

        # Find where our code ends (after calculate_stage_risk_scores check)
        our_end = code.rfind("report.append(\"\")")

        # Find where the original continues (after Section 1)
        original_start = method_code.find("# ================================================================")
        original_start = method_code.find("# SECTION 2", original_start)

        if original_start > 0:
            # Combine: our robust start + original sections 2-9 + visualizations + helper
            complete_code = (
                code[:our_end] + "\n\n        " +
                method_code[original_start:].replace("    ", "", 1)  # Dedent one level
            )

            # Now add the visualization and helper function
            viz_start = full_code.find("def create_enhanced_visualizations")
            if viz_start > 0:
                complete_code = full_code[:viz_start] + full_code[viz_start:]
        else:
            complete_code = code

        return complete_code

    except Exception as e:
        print(f"Warning: Could not read formatted file: {e}")
        return code


def update_notebook_with_robust_version(notebook_path: str) -> bool:
    """Update notebook with robust enhanced reporting."""

    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        # Find Cell 21 (EnhancedIntegrationMetrics class)
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'class EnhancedIntegrationMetrics:' in source and 'ENHANCED INTEGRATION PARADOX' in source:
                    print(f"Found EnhancedIntegrationMetrics at cell {i}")

                    # Read the complete robust version
                    with open('/home/user/L2-Research-Write-Article/formatted_enhanced_reporting.py', 'r') as f:
                        robust_code = f.read()

                    # Add the verification check at the start of __init__
                    init_marker = "def __init__(self, base_metrics):"
                    init_idx = robust_code.find(init_marker)
                    if init_idx > 0:
                        # Find the end of the docstring
                        docstring_end = robust_code.find('"""', init_idx + 50)
                        if docstring_end > 0:
                            verification_code = '''

        # Verify required methods exist
        required_methods = [
            'calculate_isolated_accuracy',
            'calculate_system_accuracy',
            'calculate_integration_gap'
        ]

        missing_methods = []
        for method in required_methods:
            if not hasattr(base_metrics, method):
                missing_methods.append(method)

        if missing_methods:
            raise AttributeError(
                f"\\n\\n{'='*70}\\n"
                f"❌ ERROR: IntegrationMetrics object is missing required methods!\\n\\n"
                f"Missing methods: {', '.join(missing_methods)}\\n\\n"
                f"💡 SOLUTION:\\n"
                f"   Make sure you run Cell 8 first, which defines the complete\\n"
                f"   IntegrationMetrics class with all required methods.\\n\\n"
                f"   Then run the PoC cells (9-17) to populate the metrics object.\\n"
                f"{'='*70}\\n"
            )
'''
                            # Insert verification after docstring
                            robust_code = (
                                robust_code[:docstring_end + 3] +
                                verification_code +
                                robust_code[docstring_end + 3:]
                            )

                    # Update the cell
                    nb['cells'][i]['source'] = robust_code.split('\n')

                    print("Updated cell with robust version (includes method verification)")
                    break

        # Write updated notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)

        print(f"\\n✅ Successfully updated notebook with robust enhanced reporting!")
        return True

    except Exception as e:
        print(f"❌ Error updating notebook: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    notebook_path = "/home/user/L2-Research-Write-Article/integration_paradox_demo.ipynb"

    print("="*80)
    print("UPDATING ENHANCED REPORTING WITH ROBUST ERROR HANDLING")
    print("="*80 + "\\n")

    success = update_notebook_with_robust_version(notebook_path)

    if success:
        print("\\n" + "="*80)
        print("✓ Update complete!")
        print("  Enhanced reporting now verifies metrics object has required methods")
        print("  Provides helpful error messages if methods are missing")
        print("="*80)
        sys.exit(0)
    else:
        print("\\n" + "="*80)
        print("✗ Update failed")
        print("="*80)
        sys.exit(1)
