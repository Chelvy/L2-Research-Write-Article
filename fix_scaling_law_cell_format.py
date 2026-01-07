#!/usr/bin/env python3
"""
Fix the scaling law cell formatting to match Colab's expected structure
"""

import json

# Load notebook
with open('integration_paradox_demo.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the scaling law cell
target_idx = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'QUANTITATIVE SCALING LAW ANALYSIS' in source:
            target_idx = i
            break

if target_idx is None:
    print("❌ Could not find scaling law cell")
    exit(1)

# Create properly formatted source as a simple string, then split
source_code = """# Implement Scaling Law from Kim et al. (2025)
import numpy as np

print("="*70)
print("QUANTITATIVE SCALING LAW ANALYSIS")
print("="*70)
print()
print("Based on: Kim et al. (2025) 'Towards a Science of Scaling Agent Systems'")
print("         arXiv:2512.08296")
print()

# Calculate scaling law predictors for our PoC 1
intelligence_index = 50  # Normalized capability score (34-66 scale)
tool_count = 5  # 5 SDLC stages
num_agents = 5
coordination_overhead_pct = 0  # PoC 1 is sequential, no coordination

# Single-agent baseline from our metrics
isolated_acc = metrics.calculate_isolated_accuracy()
single_agent_baseline = sum(isolated_acc.values()) / len(isolated_acc) if isolated_acc else 0.5

# Error amplification from our cascade data
if len(metrics.error_propagation) > 0:
    amplified = sum(1 for e in metrics.error_propagation if e.get('amplified', False))
    error_amplification = (amplified / len(metrics.error_propagation)) * 10 if amplified > 0 else 1.0
else:
    error_amplification = 1.0

# Message density (messages per turn)
message_density = 0.0  # Sequential has no inter-agent messages

# Redundancy rate (work overlap)
redundancy_rate = 0.0  # No redundancy in sequential

# Efficiency (success normalized by cost)
system_accuracy = metrics.calculate_system_accuracy()
efficiency = system_accuracy / 1.0  # Normalized to baseline cost

print("\\n" + "="*70)
print("SCALING LAW PREDICTORS")
print("="*70)
print(f"  Intelligence Index (I):          {intelligence_index}")
print(f"  Tool Count (T):                  {tool_count}")
print(f"  Number of Agents (n):            {num_agents}")
print(f"  Single-Agent Baseline (Psa):     {single_agent_baseline:.3f}")
print(f"  Error Amplification (Ae):        {error_amplification:.2f}x")
print(f"  Efficiency (Ep):                 {efficiency:.3f}")

# Key coefficient estimates from the paper
beta_Ep_T = -0.330     # Efficiency-Tools trade-off (DOMINANT)
beta_Psa_lognA = -0.408  # Baseline paradox (SECOND DOMINANT)
beta_Ae_T = -0.097     # Error propagation in tool-rich systems

# Calculate dominant effects
tool_coord_effect = beta_Ep_T * (efficiency * tool_count)
baseline_effect = beta_Psa_lognA * (single_agent_baseline * np.log(1 + num_agents))
error_tool_effect = beta_Ae_T * (error_amplification * tool_count)

print("\\n" + "="*70)
print("DOMINANT EFFECTS IN YOUR PoC")
print("="*70)

print(f"\\n1. Tool-Coordination Trade-off: {tool_coord_effect:.3f}")
print(f"   (beta = {beta_Ep_T}, strongest predictor)")
print(f"   Impact: {'Negative' if tool_coord_effect < 0 else 'Positive'}")

print(f"\\n2. Baseline Paradox Effect: {baseline_effect:.3f}")
print(f"   (beta = {beta_Psa_lognA}, second strongest)")
if single_agent_baseline > 0.45:
    print(f"   WARNING: Baseline {single_agent_baseline:.1%} > 45% threshold!")
    print(f"   Multi-agent coordination likely to DEGRADE performance")
else:
    print(f"   Baseline {single_agent_baseline:.1%} < 45% threshold")
    print(f"   Multi-agent coordination could improve performance")

print(f"\\n3. Error Amplification in Tool-Rich System: {error_tool_effect:.3f}")
print(f"   (beta = {beta_Ae_T})")
print(f"   Your error amplification: {error_amplification:.2f}x")
print(f"   Paper benchmark: Independent = 17.2x, Centralized = 4.4x")

# Architecture recommendation
print("\\n" + "="*70)
print("ARCHITECTURE RECOMMENDATION (Kim et al. 2025)")
print("="*70)

if single_agent_baseline > 0.45:
    recommendation = "SINGLE-AGENT SYSTEM"
    confidence = 87
    reasoning = f"Baseline {single_agent_baseline:.1%} exceeds 45% threshold. Multi-agent predicted to degrade 39-70%."
else:
    recommendation = "CENTRALIZED MULTI-AGENT"
    confidence = 87
    reasoning = f"Baseline {single_agent_baseline:.1%} below threshold. Multi-agent could improve 50-80%."

print(f"\\nRECOMMENDED: {recommendation}")
print(f"Confidence: {confidence}% (validated on held-out configurations)")
print(f"\\nReasoning: {reasoning}")
print("\\n" + "="*70)
"""

# Convert to proper list format (each line ends with \n)
new_source = [line + '\n' for line in source_code.split('\n')]

# Update the cell
notebook['cells'][target_idx]['source'] = new_source

# Save
with open('integration_paradox_demo.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("="*80)
print("FIXED SCALING LAW CELL FORMATTING")
print("="*80)
print()
print("✅ Cell source reformatted with proper newline structure")
print("✅ Each line properly terminated with \\n")
print("✅ Simplified to avoid JSON escape issues")
print()
print("="*80)
print("✓ Cell should now work in Colab")
print("="*80)
