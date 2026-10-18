"""Fig. 1 of the journal paper: the causal streaming chain, the two-hospital protocol and the three channel sets.
Version 1.0, 18 October 2026.
    python make_fig1_chain.py <output folder>
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'axes.titlesize': 8})
BLUE, ORANGE, GREY, GREEN = '#1f5a96', '#d0661f', '#9e9e9e', '#2e7d32'
CH18 = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ']
POS = {'FP1': (-0.31, 0.95), 'FP2': (0.31, 0.95), 'F7': (-0.81, 0.59), 'F3': (-0.41, 0.51), 'FZ': (0, 0.47),
       'F4': (0.41, 0.51), 'F8': (0.81, 0.59), 'T7': (-1, 0), 'C3': (-0.5, 0), 'CZ': (0, 0), 'C4': (0.5, 0), 'T8': (1, 0),
       'P7': (-0.81, -0.59), 'P3': (-0.41, -0.51), 'PZ': (0, -0.47), 'P4': (0.41, -0.51), 'P8': (0.81, -0.59),
       'O1': (-0.31, -0.95), 'O2': (0.31, -0.95)}


def box(ax, x, y, w, h, title, body, col, fs=6.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.003,rounding_size=0.01', fc='white', ec=col, lw=1.0))
    ax.text(x + w / 2, y + h - 0.035, title, ha='center', va='top', fontsize=6.4, fontweight='bold', color=col)
    ax.text(x + w / 2, y + h - 0.105, body, ha='center', va='top', fontsize=fs, linespacing=1.25)


def arrow(ax, a, b, col='k', lw=0.9, style='-|>', rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=7, color=col, lw=lw, connectionstyle=f'arc3,rad={rad}'))


fig = plt.figure(figsize=(7.16, 3.3))
# (a) streaming chain
ax = fig.add_axes([0.0, 0.47, 1.0, 0.53]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.text(0.005, 0.97, '(a) causal streaming chain, one recording at a time', fontsize=8, va='top')
stages = [('EDF records', '1-s data records,\n18 bipolar channels\n(Siena: formed\nfrom 19 electrodes)', '#616161'),
          ('Band-pass', '0.5-45 Hz, order 4,\nstate carried over\nblocks; Siena\n512 \u2192 256 Hz', '#616161'),
          ('4-s window', 'non-overlapping;\nlast 1024 samples\nbuffered', '#616161'),
          ('Features', '7 log band powers,\nlog line length,\nlog variance,\nkurtosis, zero-\ncrossing rate;\ncorrelations', BLUE),
          ('Patient reference', 'log features minus\nrunning median of\nthe previous 150\nwindows (10 min)', ORANGE),
          ('BRIDGE-R', 'channel graph with\nanatomical and\ncorrelation edges;\n2 message-passing\nlayers; classifier', BLUE),
          ('Alarm', '2 of the last 3\nwindows above the\nthreshold set in the\ntraining hospital', GREEN)]
n = len(stages); w, gap, y0, h = 0.12, 0.0215, 0.27, 0.56
for i, (t, b, c) in enumerate(stages):
    x = 0.004 + i * (w + gap); box(ax, x, y0, w, h, t, b, c, fs=5.9)
    if i < n - 1: arrow(ax, (x + w + 0.004, y0 + h / 2), (x + w + gap - 0.004, y0 + h / 2))
ax.text(0.5, 0.13, 'state kept by the device: filter state, last 4 s of signal, last 150 feature vectors, last 3 decisions',
        ha='center', fontsize=6.4, style='italic', color='#444444')

# (b) protocol
ax = fig.add_axes([0.0, 0.0, 0.56, 0.45]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.text(0.01, 0.98, '(b) in-hospital and cross-hospital evaluation', fontsize=8, va='top')
box(ax, 0.03, 0.16, 0.36, 0.58, 'CHB-MIT, Boston', '23 children and adolescents\n(1.5-22 years)\n682 recordings, 976.6 h\n185 seizures\nbipolar montage, 256 Hz', BLUE, fs=6.3)
box(ax, 0.61, 0.16, 0.36, 0.58, 'Siena', '14 adults (20-71 years)\n41 recordings, 141.0 h\n47 seizures\nreferential 10-20, 512 Hz', ORANGE, fs=6.3)
arrow(ax, (0.40, 0.58), (0.60, 0.58), col='k', lw=1.1); ax.text(0.5, 0.62, 'train → test', ha='center', fontsize=6.3)
arrow(ax, (0.60, 0.33), (0.40, 0.33), col='k', lw=1.1); ax.text(0.5, 0.37, 'test ← train', ha='center', fontsize=6.3)
ax.text(0.21, 0.04, 'in-hospital: 4 subject-wise folds', ha='center', fontsize=6.2, color=BLUE)
ax.text(0.79, 0.04, 'in-hospital: leave-one-subject-out', ha='center', fontsize=6.2, color=ORANGE)

# (c) channel sets
sets = [('18 channels', CH18), ('4 temporal', ['F7-T7', 'T7-P7', 'F8-T8', 'T8-P8']), ('2 temporal', ['T7-P7', 'T8-P8'])]
fig.text(0.585, 0.43, '(c) channel sets', fontsize=8, va='top')
for k, (name, chs) in enumerate(sets):
    ax = fig.add_axes([0.575 + k * 0.142, 0.02, 0.135, 0.36]); ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.35, 1.25); ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(Circle((0, 0), 1.12, fc='none', ec='#bdbdbd', lw=0.8)); ax.plot([-0.12, 0, 0.12], [1.1, 1.24, 1.1], color='#bdbdbd', lw=0.8)
    for c in CH18:
        a, b = c.split('-'); on = c in chs
        ax.plot([POS[a][0], POS[b][0]], [POS[a][1], POS[b][1]], color=BLUE if on else '#e0e0e0', lw=1.6 if on else 0.8, zorder=1)
    for e, (x, y) in POS.items(): ax.plot(x, y, 'o', ms=2.2, color='#616161', zorder=2)
    ax.text(0, -1.33, name, ha='center', fontsize=6.6)
fig.savefig(os.path.join(OUT, 'fig1_chain.png'), dpi=600, bbox_inches='tight'); fig.savefig(os.path.join(OUT, 'fig1_chain.pdf'), bbox_inches='tight')
print('fig1_chain')
