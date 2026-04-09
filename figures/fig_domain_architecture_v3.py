#!/usr/bin/env python3
"""Domain x architecture: d-prime by domain. v3.

All bars labeled. Arrow for commonsense ceiling break.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'Helvetica Neue',
    'font.size': 12,
    'axes.linewidth': 0.6,
    'axes.edgecolor': '#999999',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.color': '#666666',
    'ytick.color': '#666666',
})

SCIENCE = '#5B7E9D'
COMMON = '#9B6B9E'
BG = 'white'

models =      ['Qwen\n0.8B', 'Qwen\n2B', 'Qwen\n4B', 'Qwen\n9B', 'Gemma\nE4B', 'Gemma\nA4B']
science =     [1.535, 0.919, 1.804, 2.291, 1.724, 1.825]
commonsense = [1.431, 0.892, 1.297, 1.350, 1.862, 1.428]

x = np.arange(len(models))
width = 0.32

fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# Faint horizontal guides
for y in [0.5, 1.0, 1.5, 2.0]:
    ax.axhline(y=y, color='#eeeeee', linewidth=0.4, zorder=0)

# Separator
ax.axvline(x=3.5, color='#dddddd', linewidth=0.8, zorder=0)

# Bars
bars_s = ax.bar(x - width/2, science, width, color=SCIENCE, alpha=0.85, zorder=3)
bars_c = ax.bar(x + width/2, commonsense, width, color=COMMON, alpha=0.85, zorder=3)

# Value labels on ALL bars
for i, bar in enumerate(bars_s):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.03, f'{science[i]:.2f}',
            ha='center', va='bottom', fontsize=8.5, color=SCIENCE)
for i, bar in enumerate(bars_c):
    h = bar.get_height()
    # Bold the E4B commonsense value (the ceiling break)
    weight = 'bold' if i == 4 else 'normal'
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.03, f'{commonsense[i]:.2f}',
            ha='center', va='bottom', fontsize=8.5, color=COMMON, fontweight=weight)

# Arrow showing the commonsense ceiling break
ax.annotate('', xy=(4 + width/2, 1.862), xytext=(3 + width/2, 1.350),
            arrowprops=dict(arrowstyle='->', color=COMMON, lw=1.2,
                            connectionstyle='arc3,rad=0.3'))

# Legend (top right, no box)
ax.text(0.98, 0.97, 'science', transform=ax.transAxes, ha='right', va='top',
        fontsize=10, color=SCIENCE, fontweight='bold')
ax.text(0.98, 0.90, 'commonsense', transform=ax.transAxes, ha='right', va='top',
        fontsize=10, color=COMMON, fontweight='bold')

ax.set_xlim(-0.5, 5.5)
ax.set_ylim(0, 2.7)
ax.set_xticks(list(x))
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel("d-prime", fontsize=12, color='#333333')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/metacognition-domain-architecture-v3.png', dpi=300,
            bbox_inches='tight', facecolor=BG)
plt.savefig('../jbarnes850.github.io/assets/images/metacognition-domain-architecture.png',
            dpi=300, bbox_inches='tight', facecolor=BG)
print("Saved domain architecture v3")
