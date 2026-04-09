#!/usr/bin/env python3
"""d-prime scaling: cross-architecture comparison.

Editorial style. Single panel. Dot plot with CIs.
Story in 5 sec: E4B is highest, A4B is not where you'd expect, 2B dips.
"""

import matplotlib.pyplot as plt
import matplotlib

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

QWEN = '#5B7E9D'
GEMMA = '#D4622B'
BG = 'white'

# Data
models =  ['Qwen\n0.8B', 'Qwen\n2B', 'Qwen\n4B', 'Qwen\n9B', 'Gemma\nE4B', 'Gemma\nA4B']
d_prime = [1.549,         1.059,       1.652,       1.785,       1.818,        1.636]
ci_lo =   [1.238,         0.768,       1.413,       1.542,       1.587,        1.433]
ci_hi =   [2.174,         1.538,       1.961,       2.088,       2.093,        1.851]
colors =  [QWEN,          QWEN,        QWEN,        QWEN,        GEMMA,        GEMMA]

ci_err = [[d - lo for d, lo in zip(d_prime, ci_lo)],
          [hi - d for d, hi in zip(d_prime, ci_hi)]]

x = range(len(models))

fig, ax = plt.subplots(figsize=(8, 4))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# Horizontal reference line at zero
ax.axhline(y=0, color='#dddddd', linewidth=0.5, zorder=0)

# Faint horizontal guides
for y in [0.5, 1.0, 1.5, 2.0]:
    ax.axhline(y=y, color='#eeeeee', linewidth=0.4, zorder=0)

# Vertical separator between families
ax.axvline(x=3.5, color='#dddddd', linewidth=0.8, linestyle='-', zorder=0)

# Error bars + dots
for i in x:
    ax.errorbar(i, d_prime[i],
                yerr=[[ci_err[0][i]], [ci_err[1][i]]],
                fmt='o', color=colors[i], markersize=10,
                capsize=0, elinewidth=1.5, markeredgecolor='white',
                markeredgewidth=1.2, zorder=5)

# Value labels above each point
for i in x:
    ax.text(i, ci_hi[i] + 0.06, f'{d_prime[i]:.2f}',
            ha='center', va='bottom', fontsize=10, color=colors[i],
            fontweight='bold')

# Annotate the 2B dip
ax.annotate('worst\ndiscriminator',
            xy=(1, d_prime[1]), xytext=(1, 0.55),
            fontsize=8.5, color='#999999', ha='center',
            arrowprops=dict(arrowstyle='-', color='#cccccc', lw=1))

ax.set_xlim(-0.5, 5.5)
ax.set_ylim(0.3, 2.5)
ax.set_xticks(list(x))
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel("d-prime", fontsize=12, color='#333333')

# Only left and bottom spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/metacognition-dprime-scaling-v5.png', dpi=300, bbox_inches='tight',
            facecolor=BG)
plt.savefig('../jbarnes850.github.io/assets/images/metacognition-dprime-scaling.png',
            dpi=300, bbox_inches='tight', facecolor=BG)
print("Saved dprime scaling v5")
