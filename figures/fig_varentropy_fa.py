#!/usr/bin/env python3
"""Generate varentropy false alarm figure matching existing blog figure style."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11

# Data from varentropy_summary.json
models = ['0.8B', '2B', '4B', '9B']
fa_high_v = [0.387, 0.636, 0.365, 0.254]
fa_low_v = [0.700, 0.818, 0.608, 0.424]
spreads = [lo - hi for hi, lo in zip(fa_high_v, fa_low_v)]

x = np.arange(len(models))
width = 0.32

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [3, 2]})

# --- Left panel: grouped bar chart ---
bars_low = ax1.bar(x - width/2, [v * 100 for v in fa_low_v], width,
                   color='#e74c3c', alpha=0.85, label='Low varentropy (diffuse)')
bars_high = ax1.bar(x + width/2, [v * 100 for v in fa_high_v], width,
                    color='#2ecc71', alpha=0.85, label='High varentropy (structured)')

# Add spread annotations above the bars, centered between each pair
for i in range(len(models)):
    top = fa_low_v[i] * 100 + 4
    ax1.annotate(f'{spreads[i]*100:+.0f}pp',
                 xy=(x[i], top), fontsize=9, fontweight='bold',
                 ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='#999', alpha=0.9))

ax1.set_xlabel('Model Size (B parameters)')
ax1.set_ylabel('False alarm rate (%)')
ax1.set_title('Varentropy predicts sycophancy vulnerability',
              fontsize=12, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylim(0, 100)
ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- Right panel: V coefficient scaling ---
v_coefs = [-1.040, -1.073, -0.814, -0.737]
ax2.plot(x, v_coefs, 'o-', color='#3498db', markersize=8, linewidth=2)

ax2.axhline(y=0, color='#ccc', linewidth=0.8, linestyle='--')
ax2.fill_between(x, v_coefs, 0, alpha=0.1, color='#3498db')

ax2.set_xlabel('Model Size (B parameters)')
ax2.set_ylabel('V coefficient (logistic regression)')
ax2.set_title('Protective effect across scales',
              fontsize=12, fontweight='bold', pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.annotate('Negative = protective\n(higher V = fewer FA)',
             xy=(2.5, -0.5), fontsize=9, fontstyle='italic',
             color='#3498db', ha='center',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#eaf2f8',
                       edgecolor='none'))

plt.tight_layout(w_pad=3)
plt.savefig('../jbarnes850.github.io/assets/images/metacognition-varentropy-fa.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('figures/metacognition-varentropy-fa.png',
            dpi=200, bbox_inches='tight', facecolor='white')
print("Saved figure")
