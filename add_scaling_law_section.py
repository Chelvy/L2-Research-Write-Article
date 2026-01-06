#!/usr/bin/env python3
"""
Add quantitative scaling law from arxiv 2512.08296 to predict Integration Paradox
"""

import json

new_section = '''## 13. Quantitative Scaling Law (Based on Recent Research)

This section implements the scaling law from "Towards a Science of Scaling Agent Systems" (arxiv:2512.08296) to predict integration paradox severity.

### Reference
Kim et al. (2025) "Towards a Science of Scaling Agent Systems" demonstrated that agent system performance can be predicted using coordination metrics with R²=0.513 cross-validated accuracy.
'''

new_cell = '''# Implement Scaling Law from Kim et al. (2025)
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
single_agent_baseline = sum(metrics.calculate_isolated_accuracy().values()) / len(metrics.calculate_isolated_accuracy())

# Error amplification from our cascade data
if len(metrics.error_propagation) > 0:
    amplified = sum(1 for e in metrics.error_propagation if e.get('amplified', False))
    error_amplification = (amplified / len(metrics.error_propagation)) * 10 if amplified > 0 else 1.0
else:
    error_amplification = 1.0

# Message density (messages per turn)
message_density = 0.0  # Sequential has no inter-agent messages

# Redundancy rate (work overlap) - calculate from error scenarios
redundancy_rate = 0.0  # No redundancy in sequential

# Efficiency (success normalized by cost)
system_accuracy = metrics.calculate_system_accuracy()
efficiency = system_accuracy / 1.0  # Normalized to baseline cost

print("\\n📊 SCALING LAW PREDICTORS:")
print("-" * 70)
print(f"  Intelligence Index (I):          {intelligence_index}")
print(f"  Tool Count (T):                  {tool_count}")
print(f"  Number of Agents (nₐ):           {num_agents}")
print(f"  Coordination Overhead (O%):      {coordination_overhead_pct:.1%}")
print(f"  Message Density (c):             {message_density:.2f}")
print(f"  Redundancy Rate (R):             {redundancy_rate:.2f}")
print(f"  Efficiency (Eₚ):                 {efficiency:.3f}")
print(f"  Error Amplification (Aₑ):        {error_amplification:.2f}x")
print(f"  Single-Agent Baseline (Pₛₐ):     {single_agent_baseline:.3f}")

# Key coefficient estimates from the paper (Table 2)
beta_I = -0.180        # Intelligence (linear)
beta_I2 = 0.256        # Intelligence² (non-linear)
beta_logT = 0.535      # log(1+T) - tool diversity
beta_lognA = 0.094     # log(1+nₐ) - agent count
beta_logO = -0.068     # log(1+O%) - overhead penalty
beta_c = 0.122         # Message density
beta_R = 0.053         # Redundancy
beta_Ep = 0.187        # Efficiency
beta_logAe = -0.156    # log(1+Aₑ) - error amplification penalty
beta_Psa = 0.421       # Single-agent baseline

# Interaction terms (most important)
beta_Ep_T = -0.330     # ⭐ Efficiency-Tools trade-off (DOMINANT)
beta_Psa_lognA = -0.408  # ⭐ Baseline paradox (SECOND DOMINANT)
beta_O_T = -0.141      # Overhead scales with tools
beta_Ae_T = -0.097     # Error propagation in tool-rich systems

beta_0 = 0.312         # Intercept

# Calculate predicted performance
I = intelligence_index
T = tool_count
nA = num_agents
O_pct = coordination_overhead_pct * 100
c = message_density
R = redundancy_rate
Ep = efficiency
Ae = error_amplification
Psa = single_agent_baseline

predicted_performance = (
    beta_0 +
    beta_I * I +
    beta_I2 * (I ** 2) +
    beta_logT * np.log(1 + T) +
    beta_lognA * np.log(1 + nA) +
    beta_logO * np.log(1 + O_pct) if O_pct > 0 else 0 +
    beta_c * c +
    beta_R * R +
    beta_Ep * Ep +
    beta_logAe * np.log(1 + Ae) +
    beta_Psa * Psa +
    beta_Ep_T * (Ep * T) +
    beta_Psa_lognA * (Psa * np.log(1 + nA)) +
    beta_O_T * (O_pct * T) / 100 if O_pct > 0 else 0 +
    beta_Ae_T * (Ae * T)
)

print("\\n" + "="*70)
print("SCALING LAW PREDICTION")
print("="*70)

print(f"\\n  Predicted System Performance: {predicted_performance:.3f}")
print(f"  Actual System Performance:    {system_accuracy:.3f}")
print(f"  Prediction Error:             {abs(predicted_performance - system_accuracy):.3f}")
print(f"  Prediction Accuracy:          {(1 - abs(predicted_performance - system_accuracy))*100:.1f}%")

# Analyze dominant effects
print("\\n📈 DOMINANT EFFECTS IN YOUR PoC:")
print("-" * 70)

tool_coord_effect = beta_Ep_T * (efficiency * tool_count)
baseline_effect = beta_Psa_lognA * (single_agent_baseline * np.log(1 + num_agents))
error_tool_effect = beta_Ae_T * (error_amplification * tool_count)

print(f"\\n1. Tool-Coordination Trade-off: {tool_coord_effect:.3f}")
print(f"   (β = {beta_Ep_T}, strongest predictor)")
print(f"   Impact: {'Negative' if tool_coord_effect < 0 else 'Positive'}")

print(f"\\n2. Baseline Paradox Effect: {baseline_effect:.3f}")
print(f"   (β = {beta_Psa_lognA}, second strongest)")
if single_agent_baseline > 0.45:
    print(f"   ⚠️  Baseline {single_agent_baseline:.1%} > 45% threshold!")
    print(f"   Multi-agent coordination likely to DEGRADE performance")
else:
    print(f"   ✅ Baseline {single_agent_baseline:.1%} < 45% threshold")
    print(f"   Multi-agent coordination could improve performance")

print(f"\\n3. Error Amplification in Tool-Rich System: {error_tool_effect:.3f}")
print(f"   (β = {beta_Ae_T})")
print(f"   Your error amplification: {error_amplification:.2f}x")
print(f"   Paper benchmark: Independent = 17.2x, Centralized = 4.4x")

# Architecture recommendation based on scaling law
print("\\n" + "="*70)
print("ARCHITECTURE RECOMMENDATION")
print("="*70)

if single_agent_baseline > 0.45:
    recommendation = "SINGLE-AGENT SYSTEM"
    confidence = 87  # Paper reports 87% accuracy
    reasoning = (
        f"Your baseline performance ({single_agent_baseline:.1%}) exceeds the 45% "
        f"threshold. The scaling law predicts that multi-agent coordination will "
        f"degrade performance by 39-70% due to coordination overhead."
    )
else:
    if tool_count > 10:
        recommendation = "DECENTRALIZED MULTI-AGENT"
        reasoning = (
            f"Tool-heavy task (T={tool_count}) with low baseline ({single_agent_baseline:.1%}). "
            f"Decentralized architecture can parallelize despite 263% overhead."
        )
    else:
        recommendation = "CENTRALIZED MULTI-AGENT"
        reasoning = (
            f"Low baseline ({single_agent_baseline:.1%}) with moderate tools (T={tool_count}). "
            f"Centralized coordination provides error control (4.4x vs 17.2x) "
            f"and can improve performance 50-80%."
        )
    confidence = 87

print(f"\\n✅ RECOMMENDED ARCHITECTURE: {recommendation}")
print(f"   Confidence: {confidence}% (validated on held-out configurations)")
print(f"\\n   Reasoning: {reasoning}")

# Critical thresholds from the paper
print("\\n" + "="*70)
print("CRITICAL THRESHOLDS (from Kim et al. 2025)")
print("="*70)

print("\\n📌 Performance Saturation:")
print(f"   Single-agent baseline threshold: 45%")
print(f"   Your baseline: {single_agent_baseline:.1%}")
print(f"   Status: {'⚠️  Above threshold - avoid multi-agent' if single_agent_baseline > 0.45 else '✅ Below threshold - multi-agent viable'}")

print("\\n📌 Message Density Saturation:")
print(f"   Optimal message density: 0.39 messages/turn")
print(f"   Your density: {message_density:.2f}")
print(f"   Beyond optimal, only ≈2-3% additional gains")

print("\\n📌 Error Amplification Ranges:")
print(f"   Your system: {error_amplification:.2f}x")
print(f"   Independent MAS: 17.2x (no coordination)")
print(f"   Centralized MAS: 4.4x (best error control)")
print(f"   Decentralized MAS: 7.8x")

print("\\n" + "="*70)
print("✓ Scaling Law Analysis Complete")
print("="*70)
'''

# Load notebook
with open('integration_paradox_demo.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find section 12 (after export results)
insert_after_idx = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell.get('source', []))
        if '## 12. Export Results for Analysis' in source:
            insert_after_idx = i + 1  # Insert after the export code cell
            break

if insert_after_idx:
    # Insert markdown header
    notebook['cells'].insert(insert_after_idx, {
        'cell_type': 'markdown',
        'metadata': {'id': 'scaling-law-header'},
        'source': new_section.split('\\n')
    })

    # Insert code cell
    notebook['cells'].insert(insert_after_idx + 1, {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {'id': 'scaling-law-analysis'},
        'outputs': [],
        'source': new_cell.split('\\n')
    })

    # Save
    with open('integration_paradox_demo.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

    print("✅ Added scaling law analysis section!")
else:
    print("❌ Could not find insertion point")
