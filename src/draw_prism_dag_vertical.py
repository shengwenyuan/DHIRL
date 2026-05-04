#!/usr/bin/env python3
"""
PRISM Algorithm Pipeline DAG — Vertical Layout
Publication-quality figure for top-conference papers/posters.
Outputs: figs/prism_dag_vertical.png  (300 DPI)
         figs/prism_dag_vertical.pdf  (vector, for embedding in LaTeX)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.lines import Line2D

os.makedirs('figs', exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────
C = dict(
    # sequence-model variants
    rnn_face  = '#DBEAFE', rnn_edge  = '#1D4ED8', rnn_text  = '#1E3A8A',
    lstm_face = '#FEF3C7', lstm_edge = '#D97706', lstm_text = '#92400E',
    tf_face   = '#D1FAE5', tf_edge   = '#059669', tf_text   = '#065F46',
    # data nodes
    data_face = '#F1F5F9', data_edge = '#64748B', data_text = '#0F172A',
    # generic process nodes
    proc_face = '#FFFFFF', proc_edge = '#374151', proc_text = '#111827',
    # specialised nodes
    intnet_face = '#F8FAFC', intnet_edge = '#94A3B8',
    policy_face = '#EFF6FF', policy_edge = '#3B82F6', policy_text = '#1E40AF',
    combine_face= '#F5F3FF', combine_edge= '#7C3AED', combine_text= '#4C1D95',
    post_face   = '#FEFCE8', post_edge   = '#CA8A04', post_text   = '#713F12',
    out_face    = '#F0FDF4', out_edge    = '#16A34A', out_text    = '#14532D',
    # section backgrounds
    e_bg  = '#EFF6FF', m_bg  = '#FFF7ED',
    em_edge = '#4B5563',
    # arrows
    arrow      = '#1F2937',
    arrow_dash = '#94A3B8',
    arrow_soft = '#CBD5E1',
)

# ── Figure canvas ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.2, 15.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 22.5)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')

FF = 'DejaVu Sans'   # font family


# ═══════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════

def rounded_node(cx, cy, w, h, text, sub=None,
                 fc=C['proc_face'], ec=C['proc_edge'], tc=C['proc_text'],
                 lw=1.8, alpha=1.0, zorder=5, fs=10, fw='bold', pad=0.10):
    """Draw a rounded-rectangle node and return its edge midpoints."""
    patch = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f'round,pad={pad}',
        facecolor=fc, edgecolor=ec,
        linewidth=lw, alpha=alpha, zorder=zorder, clip_on=False,
    )
    ax.add_patch(patch)
    if sub:
        ax.text(cx, cy + 0.17, text, ha='center', va='center',
                fontsize=fs, fontweight=fw, color=tc, zorder=zorder+1, family=FF)
        ax.text(cx, cy - 0.19, sub, ha='center', va='center',
                fontsize=fs - 1.5, color=tc, zorder=zorder+1, family=FF,
                fontstyle='italic')
    else:
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=fs, fontweight=fw, color=tc, zorder=zorder+1, family=FF)
    return dict(top=(cx, cy+h/2), bot=(cx, cy-h/2),
                left=(cx-w/2, cy), right=(cx+w/2, cy))


def para_node(cx, cy, w, h, text, sub=None,
              fc=C['data_face'], ec=C['data_edge'], tc=C['data_text'],
              lw=1.8, zorder=5, fs=10, skew=0.28, alpha=1.0):
    """Draw a parallelogram (data node)."""
    d = skew
    xs = [cx-w/2+d, cx+w/2+d, cx+w/2-d, cx-w/2-d]
    ys = [cy-h/2,   cy-h/2,   cy+h/2,   cy+h/2  ]
    ax.add_patch(Polygon(list(zip(xs, ys)), closed=True,
                         facecolor=fc, edgecolor=ec,
                         linewidth=lw, zorder=zorder, alpha=alpha))
    if sub:
        ax.text(cx, cy+0.17, text, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=tc, zorder=zorder+1, family=FF)
        ax.text(cx, cy-0.19, sub, ha='center', va='center',
                fontsize=fs-1.5, color=tc, zorder=zorder+1, family=FF,
                fontstyle='italic')
    else:
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=tc, zorder=zorder+1, family=FF)
    return dict(top=(cx, cy+h/2), bot=(cx, cy-h/2))


def diamond_node(cx, cy, w, h, text,
                 fc='#F9FAFB', ec='#374151', tc='#111827',
                 lw=1.8, zorder=5, fs=9):
    """Draw a diamond decision node."""
    xs = [cx,     cx+w/2, cx,     cx-w/2]
    ys = [cy+h/2, cy,     cy-h/2, cy    ]
    ax.add_patch(Polygon(list(zip(xs, ys)), closed=True,
                         facecolor=fc, edgecolor=ec,
                         linewidth=lw, zorder=zorder))
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, fontweight='bold', color=tc,
            zorder=zorder+1, family=FF)


def section_bg(x0, y0, w, h, fc, alpha=0.42, lw=0, ec='none', zorder=1,
               label=None, lx=None, ly=None,
               lfs=9.5, lfc='#374151', lfw='bold', lfs_italic=True):
    """Draw a tinted background region with optional section label."""
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle='round,pad=0.08',
        facecolor=fc, edgecolor=ec,
        linewidth=lw, alpha=alpha, zorder=zorder, clip_on=False,
    ))
    if label:
        ax.text(lx if lx else x0+0.18,
                ly if ly else y0+h-0.28,
                label,
                ha='left', va='top',
                fontsize=lfs, fontweight=lfw, color=lfc,
                fontstyle='italic' if lfs_italic else 'normal',
                zorder=zorder+1, family=FF)


def arrow(xy_s, xy_e, label='', dashed=False, color=C['arrow'],
          lw=1.55, rad=0.0, lfs=8.0, lcolor='#4B5563', loffset=(0.12, 0.0),
          ha='left', zorder=20):
    """Draw a directed arrow between two points."""
    ls = (0, (5, 4)) if dashed else '-'
    ax.annotate('', xy=xy_e, xytext=xy_s,
                arrowprops=dict(
                    arrowstyle='->', color=color, lw=lw,
                    linestyle=ls,
                    connectionstyle=f'arc3,rad={rad}',
                    shrinkA=2, shrinkB=2,
                ),
                zorder=zorder)
    if label:
        mx = (xy_s[0]+xy_e[0])/2 + loffset[0]
        my = (xy_s[1]+xy_e[1])/2 + loffset[1]
        ax.text(mx, my, label, ha=ha, va='center',
                fontsize=lfs, color=lcolor, zorder=zorder+1, family=FF,
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.85))


def angle_arrow(xy_s, xy_e, label='', dashed=False, color=C['arrow'],
                lw=1.55, angA=0, angB=90, rad=0.25, lfs=8.0,
                lcolor='#4B5563', loffset=(0.12, 0.0), zorder=20):
    """Draw an L-shaped / angled arrow (good for routing around boxes)."""
    ls = (0, (5, 4)) if dashed else '-'
    ax.annotate('', xy=xy_e, xytext=xy_s,
                arrowprops=dict(
                    arrowstyle='->', color=color, lw=lw,
                    linestyle=ls,
                    connectionstyle=f'angle,angleA={angA},angleB={angB},rad={rad}',
                    shrinkA=2, shrinkB=2,
                ),
                zorder=zorder)
    if label:
        mx = (xy_s[0]+xy_e[0])/2 + loffset[0]
        my = (xy_s[1]+xy_e[1])/2 + loffset[1]
        ax.text(mx, my, label, ha='left', va='center',
                fontsize=lfs, color=lcolor, zorder=zorder+1, family=FF,
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.85))


# ═══════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════
ax.text(5.0, 22.15, 'PRISM', ha='center', va='center',
        fontsize=20, fontweight='bold', color='#0F172A', family=FF, zorder=30)
ax.text(5.0, 21.72, 'Algorithm Pipeline', ha='center', va='center',
        fontsize=9.5, color='#64748B', fontstyle='italic', family=FF, zorder=30)

# thin separator line under title
ax.plot([1.5, 8.5], [21.45, 21.45], color='#CBD5E1', lw=1.0, zorder=29)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION BACKGROUNDS  (drawn first — lowest z-order)
# ═══════════════════════════════════════════════════════════════════════════

# EM Loop outer dashed border
ax.add_patch(FancyBboxPatch(
    (0.38, 1.15), 9.24, 16.45,
    boxstyle='round,pad=0.10',
    facecolor='none', edgecolor=C['em_edge'],
    linewidth=1.7, linestyle=(0, (6, 3)),
    zorder=2, clip_on=False,
))
ax.text(9.38, 17.52, 'EM Loop', ha='right', va='center',
        fontsize=8.5, fontweight='bold', color=C['em_edge'],
        fontstyle='italic', family=FF, zorder=3)

# E-Step band
section_bg(0.52, 9.62, 8.96, 7.88, C['e_bg'], zorder=1,
           label='E-Step', lfc='#1D4ED8', lx=0.70, ly=17.42)

# M-Step band
section_bg(0.52, 3.28, 8.96, 6.05, C['m_bg'], zorder=1,
           label='M-Step', lfc='#B45309', lx=0.70, ly=9.25)

# Intention Network inner box
ax.add_patch(FancyBboxPatch(
    (5.22, 11.08), 4.42, 6.02,
    boxstyle='round,pad=0.09',
    facecolor=C['intnet_face'], edgecolor=C['intnet_edge'],
    linewidth=1.3, alpha=0.92, zorder=3, clip_on=False,
))
ax.text(7.43, 16.92, 'Intention Network', ha='center', va='center',
        fontsize=9.0, fontweight='bold', color='#334155',
        family=FF, zorder=4)


# ═══════════════════════════════════════════════════════════════════════════
# NODES
# ═══════════════════════════════════════════════════════════════════════════

# ── INPUT (above EM loop) ──────────────────────────────────────────────────
para_node(5.0, 20.95, 5.4, 0.85,
          'Expert Demonstrations',
          sub='trajectories  { (sₜ, aₜ) }',
          fc=C['data_face'], ec=C['data_edge'], tc=C['data_text'],
          zorder=5, fs=10)

rounded_node(5.0, 19.40, 5.5, 0.88,
             'Trajectory Encoder',
             sub='batch_states,  batch_actions,  batch_mask',
             fc=C['proc_face'], ec=C['proc_edge'], tc=C['proc_text'],
             zorder=5, fs=10)

rounded_node(5.0, 17.80, 5.5, 0.88,
             'Initialize  K  Agents',
             sub='K × IAVI_B with uniform policy',
             fc=C['proc_face'], ec=C['proc_edge'], tc=C['proc_text'],
             zorder=5, fs=10)

# ── E-STEP ─────────────────────────────────────────────────────────────────
rounded_node(2.65, 14.72, 4.08, 0.88,
             'Agent Policies  πₖ',
             sub='softmax(Qₖ),   k = 1 … K',
             fc=C['policy_face'], ec=C['policy_edge'], tc=C['policy_text'],
             zorder=5, fs=10, lw=1.7)

# — Intention Network internals —
rounded_node(7.43, 15.72, 3.58, 0.74,
             'State & Action Embedding',
             sub='Embed(sₜ) ⊕ Embed(aₜ)',
             fc='#F8FAFC', ec='#94A3B8', tc='#334155',
             zorder=6, fs=9, lw=1.3)

# Stacked sequence-model blocks (back → front = Transformer → LSTM → RNN)
SW, SH = 3.20, 0.76           # block size
SCX, SCY = 7.43, 14.10        # front (RNN) centre
OFF = 0.18                     # spatial offset per layer

# Transformer — back, green
rounded_node(SCX + 2*OFF, SCY - 2*OFF, SW, SH,
             'Transformer',
             fc=C['tf_face'], ec=C['tf_edge'], tc=C['tf_text'],
             alpha=0.55, zorder=7, fs=9, lw=1.9)

# LSTM — middle, amber
rounded_node(SCX + OFF, SCY - OFF, SW, SH,
             'LSTM',
             fc=C['lstm_face'], ec=C['lstm_edge'], tc=C['lstm_text'],
             alpha=0.58, zorder=8, fs=9, lw=1.9)

# RNN — front, blue (default)
rounded_node(SCX, SCY, SW, SH,
             'RNN  ★',
             fc=C['rnn_face'], ec=C['rnn_edge'], tc=C['rnn_text'],
             alpha=1.0, zorder=9, fs=9.5, lw=2.1)

ax.text(SCX, SCY - SH/2 - 0.23, '★ default',
        ha='center', va='top', fontsize=7.5,
        color='#64748B', fontstyle='italic', family=FF, zorder=10)

rounded_node(7.43, 12.50, 3.58, 0.74,
             'Output Projection',
             sub='Linear  →  logits  (B, T, K)',
             fc='#F8FAFC', ec='#94A3B8', tc='#334155',
             zorder=6, fs=9, lw=1.3)

# — Combine ―
rounded_node(5.0, 10.70, 5.55, 0.88,
             'Log-Space Fusion',
             sub='log f (s,a)  +  log πₖ(a|s)',
             fc=C['combine_face'], ec=C['combine_edge'], tc=C['combine_text'],
             zorder=5, fs=10, lw=1.8)

# — Intent Posterior ―
rounded_node(5.0, 9.12, 5.55, 0.88,
             'Intent Posterior  γₜ,ₖ',
             sub='P(zₜ = k | trajectory)',
             fc=C['post_face'], ec=C['post_edge'], tc=C['post_text'],
             zorder=5, fs=10, lw=1.8)

# ── M-STEP ─────────────────────────────────────────────────────────────────
rounded_node(2.65, 6.62, 4.08, 0.88,
             'Update  K  Agents',
             sub='IAVI_B × K,  weighted by  γ',
             fc=C['proc_face'], ec='#4B5563', tc=C['proc_text'],
             zorder=5, fs=10, lw=1.6)

rounded_node(7.35, 6.62, 4.08, 0.88,
             'Train  Intention  Net',
             sub='ℒ = NLL + λ₁ L1 + λ₂ KL',
             fc=C['proc_face'], ec='#4B5563', tc=C['proc_text'],
             zorder=5, fs=10, lw=1.6)

diamond_node(5.0, 4.70, 2.85, 1.15, 'Converged?',
             fc='#F9FAFB', ec='#374151', tc='#111827',
             lw=1.8, zorder=5, fs=9)

# ── OUTPUT (below EM loop) ─────────────────────────────────────────────────
# Double-border effect: draw outer then inner parallelogram
para_node(5.0, 0.82, 5.75, 0.88,
          'Output',
          sub='K  Intent Policies  +  Reward Functions',
          fc=C['out_face'], ec=C['out_edge'], tc=C['out_text'],
          zorder=5, fs=10, skew=0.30)
para_node(5.0, 0.82, 5.35, 0.72, '',
          fc='none', ec=C['out_edge'],
          zorder=6, fs=10, skew=0.26)


# ═══════════════════════════════════════════════════════════════════════════
# ARROWS
# ═══════════════════════════════════════════════════════════════════════════

# Expert Demos → Traj Encoder
arrow((5.0, 20.525), (5.0, 19.845))

# Traj Encoder → Init K Agents
arrow((5.0, 18.955), (5.0, 18.245))

# Init K Agents → Agent Policies  (diagonal, entering E-step left)
arrow((3.05, 17.355), (2.65, 15.165),
      label='init agents', loffset=(0.12, 0.08), lfs=7.5, ha='left')

# Traj Encoder → Embed  (curves right, avoids Init K Agents)
angle_arrow((7.55, 18.955), (7.43, 16.095),
            angA=0, angB=90, rad=0.3,
            label='encoded trajs', loffset=(-0.55, 0.2), lfs=7.5)

# Agent Policies (right side) → Combine (left side)  [diagonal]
arrow((4.69, 14.72), (3.55, 11.145),
      label='log πₖ', loffset=(-1.05, 0.05), lfs=7.5, ha='right')

# Internal: Embed → RNN front-face top
arrow((7.43, 15.355), (7.43, 14.485), lw=1.3, color='#475569')

# Internal: RNN front-face bottom → Output Proj top
arrow((7.43, 13.715), (7.43, 12.875), lw=1.3, color='#475569')

# Output Proj bottom → Combine right
arrow((7.10, 12.125), (7.25, 11.145))

# Combine → Intent Posterior
arrow((5.0, 10.255), (5.0, 9.565))

# Intent Posterior → Update K Agents  (bottom-left diagonal)
arrow((3.20, 8.675), (2.65, 7.065),
      label='γ', loffset=(-0.35, 0.06), lfs=9.5, ha='right')

# Intent Posterior → Train Intention Net  (bottom-right diagonal)
arrow((6.80, 8.675), (7.35, 7.065),
      label='γ', loffset=(0.10, 0.06), lfs=9.5)

# Update K Agents → Convergence
arrow((3.25, 6.175), (4.08, 5.260))

# Train Intention Net → Convergence
arrow((6.75, 6.175), (5.92, 5.260))

# Convergence → Output  ("yes" branch, straight down)
arrow((5.0, 4.145), (5.0, 1.265),
      label='yes', loffset=(0.12, 0.0), lfs=8.0)

# ── Feedback loop: "no / iterate" (Convergence → E-Step) ──
# L-shaped: go left from Convergence, rise along EM loop left border, arrive at E-step
angle_arrow((3.58, 4.70), (0.85, 14.72),
            dashed=True, color=C['arrow_dash'],
            angA=180, angB=270, rad=0.30, lw=1.5)
ax.text(0.62, 9.85, 'iterate', ha='center', va='center',
        fontsize=8.0, color='#94A3B8', fontstyle='italic',
        rotation=90, family=FF, zorder=25)
ax.text(3.30, 3.95, 'no', ha='center', va='center',
        fontsize=8.0, color='#94A3B8', family=FF, zorder=25)

# ── Soft feedback inside EM loop (Updated agents / net back to E-step) ──
# Updated agents → Agent Policies (next iteration, soft dashed)
arrow((2.65, 6.175), (2.65, 15.165),
      dashed=True, color=C['arrow_soft'], lw=1.0, rad=-0.55)

# Updated intention net → Embed (next iteration, soft dashed)
arrow((7.35, 6.175), (7.43, 16.095),
      dashed=True, color=C['arrow_soft'], lw=1.0, rad=0.55)


# ═══════════════════════════════════════════════════════════════════════════
# SEQUENCE-MODEL LEGEND
# ═══════════════════════════════════════════════════════════════════════════
lx, ly = 0.65, 8.35
ax.text(lx, ly, 'Seq. Model:', ha='left', va='center',
        fontsize=8, fontweight='bold', color='#374151', family=FF, zorder=25)

swatches = [
    (lx + 1.65, C['rnn_face'],  C['rnn_edge'],  C['rnn_text'],  'RNN (default)', 1.0),
    (lx + 4.10, C['lstm_face'], C['lstm_edge'], C['lstm_text'], 'LSTM',          0.75),
    (lx + 6.20, C['tf_face'],   C['tf_edge'],   C['tf_text'],   'Transformer',   0.75),
]
for sx, sf, se, st, slabel, salpha in swatches:
    ax.add_patch(FancyBboxPatch(
        (sx, ly - 0.26), 0.42, 0.30,
        boxstyle='round,pad=0.03',
        facecolor=sf, edgecolor=se,
        linewidth=1.4, alpha=salpha, zorder=25,
    ))
    ax.text(sx + 0.55, ly - 0.11, slabel,
            ha='left', va='center', fontsize=8,
            color=st, family=FF, zorder=25)


# ═══════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.3)
plt.savefig('figs/prism_dag_vertical.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figs/prism_dag_vertical.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('Saved  figs/prism_dag_vertical.{png,pdf}')
plt.close()
