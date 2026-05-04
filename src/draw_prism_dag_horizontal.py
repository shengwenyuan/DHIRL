#!/usr/bin/env python3
"""
PRISM Algorithm Pipeline DAG — Horizontal Layout
Publication-quality figure for top-conference papers/posters.
Outputs: figs/prism_dag_horizontal.png  (300 DPI)
         figs/prism_dag_horizontal.pdf  (vector, for embedding in LaTeX)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon

os.makedirs('figs', exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────
C = dict(
    rnn_face  = '#DBEAFE', rnn_edge  = '#1D4ED8', rnn_text  = '#1E3A8A',
    lstm_face = '#FEF3C7', lstm_edge = '#D97706', lstm_text = '#92400E',
    tf_face   = '#D1FAE5', tf_edge   = '#059669', tf_text   = '#065F46',
    data_face = '#F1F5F9', data_edge = '#64748B', data_text = '#0F172A',
    proc_face = '#FFFFFF', proc_edge = '#374151', proc_text = '#111827',
    intnet_face = '#F8FAFC', intnet_edge = '#94A3B8',
    policy_face = '#EFF6FF', policy_edge = '#3B82F6', policy_text = '#1E40AF',
    combine_face= '#F5F3FF', combine_edge= '#7C3AED', combine_text= '#4C1D95',
    post_face   = '#FEFCE8', post_edge   = '#CA8A04', post_text   = '#713F12',
    out_face    = '#F0FDF4', out_edge    = '#16A34A', out_text    = '#14532D',
    e_bg  = '#EFF6FF', m_bg  = '#FFF7ED',
    em_edge = '#4B5563',
    arrow      = '#1F2937',
    arrow_dash = '#94A3B8',
    arrow_soft = '#CBD5E1',
)

# ── Figure canvas ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 8.5))
ax.set_xlim(0, 26)
ax.set_ylim(0, 9.5)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')

FF = 'DejaVu Sans'


# ═══════════════════════════════════════════════════════════════════════════
# Drawing helpers  (shared with vertical script)
# ═══════════════════════════════════════════════════════════════════════════

def rounded_node(cx, cy, w, h, text, sub=None,
                 fc=C['proc_face'], ec=C['proc_edge'], tc=C['proc_text'],
                 lw=1.8, alpha=1.0, zorder=5, fs=10, fw='bold', pad=0.10):
    patch = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f'round,pad={pad}',
        facecolor=fc, edgecolor=ec,
        linewidth=lw, alpha=alpha, zorder=zorder, clip_on=False,
    )
    ax.add_patch(patch)
    if sub:
        ax.text(cx, cy + 0.16, text, ha='center', va='center',
                fontsize=fs, fontweight=fw, color=tc, zorder=zorder+1, family=FF)
        ax.text(cx, cy - 0.18, sub, ha='center', va='center',
                fontsize=fs - 1.5, color=tc, zorder=zorder+1, family=FF,
                fontstyle='italic')
    else:
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=fs, fontweight=fw, color=tc, zorder=zorder+1, family=FF)


def para_node(cx, cy, w, h, text, sub=None,
              fc=C['data_face'], ec=C['data_edge'], tc=C['data_text'],
              lw=1.8, zorder=5, fs=10, skew=0.22, alpha=1.0):
    d = skew
    xs = [cx-w/2+d, cx+w/2+d, cx+w/2-d, cx-w/2-d]
    ys = [cy-h/2,   cy-h/2,   cy+h/2,   cy+h/2  ]
    ax.add_patch(Polygon(list(zip(xs, ys)), closed=True,
                         facecolor=fc, edgecolor=ec,
                         linewidth=lw, zorder=zorder, alpha=alpha))
    if sub:
        ax.text(cx, cy+0.16, text, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=tc, zorder=zorder+1, family=FF)
        ax.text(cx, cy-0.18, sub, ha='center', va='center',
                fontsize=fs-1.5, color=tc, zorder=zorder+1, family=FF,
                fontstyle='italic')
    else:
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=tc, zorder=zorder+1, family=FF)


def diamond_node(cx, cy, w, h, text,
                 fc='#F9FAFB', ec='#374151', tc='#111827',
                 lw=1.8, zorder=5, fs=9):
    xs = [cx,     cx+w/2, cx,     cx-w/2]
    ys = [cy+h/2, cy,     cy-h/2, cy    ]
    ax.add_patch(Polygon(list(zip(xs, ys)), closed=True,
                         facecolor=fc, edgecolor=ec,
                         linewidth=lw, zorder=zorder))
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, fontweight='bold', color=tc,
            zorder=zorder+1, family=FF)


def section_bg(x0, y0, w, h, fc, alpha=0.42, lw=0, ec='none', zorder=1,
               label=None, lx=None, ly=None, lfs=9.5, lfc='#374151'):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle='round,pad=0.08',
        facecolor=fc, edgecolor=ec,
        linewidth=lw, alpha=alpha, zorder=zorder, clip_on=False,
    ))
    if label:
        ax.text(lx if lx else x0+0.2,
                ly if ly else y0+h-0.25,
                label, ha='left', va='top',
                fontsize=lfs, fontweight='bold', color=lfc,
                fontstyle='italic', zorder=zorder+1, family=FF)


def arrow(xy_s, xy_e, label='', dashed=False, color=C['arrow'],
          lw=1.55, rad=0.0, lfs=8.0, lcolor='#4B5563', loffset=(0.0, 0.15),
          ha='center', zorder=20):
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
                lw=1.55, angA=0, angB=90, rad=0.25,
                lfs=8.0, lcolor='#4B5563', loffset=(0.0, 0.15), ha='center',
                zorder=20):
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
        ax.text(mx, my, label, ha=ha, va='center',
                fontsize=lfs, color=lcolor, zorder=zorder+1, family=FF,
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.85))


# ═══════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════
ax.text(13.0, 9.22, 'PRISM', ha='center', va='center',
        fontsize=20, fontweight='bold', color='#0F172A', family=FF, zorder=30)
ax.text(13.0, 8.85, 'Algorithm Pipeline', ha='center', va='center',
        fontsize=9.5, color='#64748B', fontstyle='italic', family=FF, zorder=30)
ax.plot([4.0, 22.0], [8.62, 8.62], color='#CBD5E1', lw=1.0, zorder=29)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION BACKGROUNDS
# ═══════════════════════════════════════════════════════════════════════════

# EM Loop outer dashed border
ax.add_patch(FancyBboxPatch(
    (5.35, 0.28), 19.8, 8.05,
    boxstyle='round,pad=0.10',
    facecolor='none', edgecolor=C['em_edge'],
    linewidth=1.7, linestyle=(0, (6, 3)),
    zorder=2, clip_on=False,
))
ax.text(24.95, 8.25, 'EM Loop', ha='right', va='center',
        fontsize=8.5, fontweight='bold', color=C['em_edge'],
        fontstyle='italic', family=FF, zorder=3)

# E-Step band (upper)
section_bg(5.45, 4.55, 19.6, 3.75, C['e_bg'], zorder=1,
           label='E-Step', lfc='#1D4ED8', lx=5.65, ly=8.22)

# M-Step band (lower)
section_bg(5.45, 0.38, 19.6, 3.95, C['m_bg'], zorder=1,
           label='M-Step', lfc='#B45309', lx=5.65, ly=4.25)

# Intention Network inner box
ax.add_patch(FancyBboxPatch(
    (9.72, 4.82), 6.45, 3.05,
    boxstyle='round,pad=0.09',
    facecolor=C['intnet_face'], edgecolor=C['intnet_edge'],
    linewidth=1.3, alpha=0.92, zorder=3, clip_on=False,
))
ax.text(12.95, 7.72, 'Intention Network', ha='center', va='center',
        fontsize=9.0, fontweight='bold', color='#334155',
        family=FF, zorder=4)


# ═══════════════════════════════════════════════════════════════════════════
# NODES
# ═══════════════════════════════════════════════════════════════════════════

# ── INPUT (left, outside EM loop) ─────────────────────────────────────────
para_node(2.2, 4.5, 2.6, 0.85,
          'Expert Demo.',
          sub='{ (sₜ, aₜ) }',
          fc=C['data_face'], ec=C['data_edge'], tc=C['data_text'],
          zorder=5, fs=9.5, skew=0.22)

rounded_node(4.7, 6.9, 2.8, 0.85,
             'Trajectory\nEncoder',
             fc=C['proc_face'], ec=C['proc_edge'], tc=C['proc_text'],
             zorder=5, fs=9.5)

rounded_node(4.7, 2.1, 2.8, 0.85,
             'Initialize\nK  Agents',
             sub='K × IAVI_B',
             fc=C['proc_face'], ec=C['proc_edge'], tc=C['proc_text'],
             zorder=5, fs=9.5)

# ── E-STEP ─────────────────────────────────────────────────────────────────

# Agent Policies (upper-left of E-step, same height as Combine so arrow is straight)
rounded_node(8.2, 7.92, 3.2, 0.85,
             'Agent Policies  πₖ',
             sub='softmax(Qₖ),  k=1…K',
             fc=C['policy_face'], ec=C['policy_edge'], tc=C['policy_text'],
             zorder=5, fs=9.5, lw=1.7)

# Intention Network internals
rounded_node(12.95, 7.30, 5.5, 0.70,
             'State & Action Embedding',
             sub='Embed(sₜ) ⊕ Embed(aₜ)',
             fc='#F8FAFC', ec='#94A3B8', tc='#334155',
             zorder=6, fs=8.5, lw=1.3)

SW, SH = 4.7, 0.68
SCX, SCY = 12.95, 6.25
OFF = 0.16

# Transformer (back, green)
rounded_node(SCX + 2*OFF, SCY - 2*OFF, SW, SH,
             'Transformer',
             fc=C['tf_face'], ec=C['tf_edge'], tc=C['tf_text'],
             alpha=0.55, zorder=7, fs=9, lw=1.9)

# LSTM (middle, amber)
rounded_node(SCX + OFF, SCY - OFF, SW, SH,
             'LSTM',
             fc=C['lstm_face'], ec=C['lstm_edge'], tc=C['lstm_text'],
             alpha=0.58, zorder=8, fs=9, lw=1.9)

# RNN (front, blue)
rounded_node(SCX, SCY, SW, SH,
             'RNN  ★',
             fc=C['rnn_face'], ec=C['rnn_edge'], tc=C['rnn_text'],
             alpha=1.0, zorder=9, fs=9.5, lw=2.1)

ax.text(SCX, SCY - SH/2 - 0.18, '★ default',
        ha='center', va='top', fontsize=7.5,
        color='#64748B', fontstyle='italic', family=FF, zorder=10)

rounded_node(12.95, 5.22, 5.5, 0.70,
             'Output Projection',
             sub='Linear  →  logits  (B, T, K)',
             fc='#F8FAFC', ec='#94A3B8', tc='#334155',
             zorder=6, fs=8.5, lw=1.3)

# Combine (upper-right of E-step, same y as Agent Policies → horizontal arrow clears Int Net)
rounded_node(18.7, 7.92, 3.2, 0.85,
             'Log-Space Fusion',
             sub='log f(s,a)  +  log πₖ(a|s)',
             fc=C['combine_face'], ec=C['combine_edge'], tc=C['combine_text'],
             zorder=5, fs=9.5, lw=1.8)

# Intent Posterior (bridges E and M-step at right)
rounded_node(22.5, 6.35, 3.5, 0.85,
             'Intent Posterior  γₜ,ₖ',
             sub='P(zₜ = k | trajectory)',
             fc=C['post_face'], ec=C['post_edge'], tc=C['post_text'],
             zorder=5, fs=9.5, lw=1.8)

# ── M-STEP ─────────────────────────────────────────────────────────────────
rounded_node(10.8, 2.8, 3.8, 0.85,
             'Update  K  Agents',
             sub='IAVI_B × K,  weighted by  γ',
             fc=C['proc_face'], ec='#4B5563', tc=C['proc_text'],
             zorder=5, fs=9.5, lw=1.6)

rounded_node(17.5, 2.8, 3.8, 0.85,
             'Train  Intention  Net',
             sub='ℒ = NLL + λ₁ L1 + λ₂ KL',
             fc=C['proc_face'], ec='#4B5563', tc=C['proc_text'],
             zorder=5, fs=9.5, lw=1.6)

diamond_node(22.5, 1.85, 2.7, 1.1, 'Converged?',
             fc='#F9FAFB', ec='#374151', tc='#111827',
             lw=1.8, zorder=5, fs=9)

# ── OUTPUT (right, outside EM loop) ───────────────────────────────────────
para_node(25.4, 4.5, 2.85, 0.85,
          'Output',
          sub='K  Policies  +  Rewards',
          fc=C['out_face'], ec=C['out_edge'], tc=C['out_text'],
          zorder=5, fs=9.5, skew=0.22)
para_node(25.4, 4.5, 2.50, 0.70, '',
          fc='none', ec=C['out_edge'],
          zorder=6, fs=9.5, skew=0.18)


# ═══════════════════════════════════════════════════════════════════════════
# ARROWS
# ═══════════════════════════════════════════════════════════════════════════

# Expert Demo → Traj Encoder  (upper diagonal)
arrow((3.5, 4.92), (3.3, 6.47))

# Expert Demo → Init K Agents  (lower diagonal)
arrow((3.5, 4.08), (3.3, 2.53))

# Traj Encoder → enters EM loop → Embed top  (exit right, arrive from above)
angle_arrow((6.1, 6.9), (12.95, 7.65),
            angA=0, angB=90, rad=0.3,
            label='encoded trajs', loffset=(0.0, 0.20), lfs=7.5)

# Init K Agents → enters EM loop → Agent Policies  (arc up the left side of E-step)
arrow((6.1, 2.1), (6.6, 7.49),
      rad=0.28,
      label='init agents', loffset=(-1.4, 0.0), lfs=7.5, lcolor='#4B5563', ha='right')

# Agent Policies → Combine  (straight horizontal, clears Int Net box top at y=7.87)
arrow((9.8, 7.92), (17.1, 7.92),
      label='log πₖ', loffset=(0.0, 0.22), lfs=7.5)

# Internal: Embed → RNN front top
arrow((12.95, 6.95), (12.95, 6.59), lw=1.3, color='#475569')

# Internal: RNN front bottom → Output Proj top
arrow((12.95, 5.91), (12.95, 5.57), lw=1.3, color='#475569')

# Output Proj (right of Int Net box) → Combine bottom
angle_arrow((16.17, 5.22), (17.1, 7.49),
            angA=0, angB=90, rad=0.25)

# Combine → Intent Posterior  (diagonal down-right)
arrow((20.3, 7.92), (20.75, 6.78),
      label='normalize', loffset=(0.18, 0.0), lfs=7.5)

# Intent Posterior → Update K Agents  (diagonal down-left, long)
arrow((21.0, 5.92), (12.7, 3.23),
      label='γ', loffset=(0.0, 0.20), lfs=9.5, lcolor=C['post_edge'])

# Intent Posterior → Train Intent Net  (diagonal down-left, medium)
arrow((24.0, 5.92), (19.4, 3.23),
      label='γ', loffset=(0.12, 0.0), lfs=9.5, lcolor=C['post_edge'])

# Update K Agents → Convergence
arrow((12.7, 2.8), (21.15, 2.17), rad=0.0)

# Train Intent Net → Convergence
arrow((19.4, 2.8), (21.15, 2.27), rad=0.15)

# Convergence → Output  (curve right and up)
arrow((23.85, 1.85), (24.0, 4.09),
      rad=-0.45,
      label='yes', loffset=(0.2, 0.0), lfs=8.0)

# Feedback: Convergence → E-step (not converged) — arc over the top
arrow((22.5, 2.41), (7.6, 7.92),
      dashed=True, color=C['arrow_dash'], lw=1.5,
      rad=-0.38,
      label='no / iterate', loffset=(0.0, 0.25), lfs=8.0, lcolor='#9CA3AF')

# Soft feedback: updated agents feed back to next E-step iteration
arrow((8.9, 2.8), (6.8, 7.49),
      dashed=True, color=C['arrow_soft'], lw=1.0, rad=-0.45)
arrow((15.6, 2.8), (12.95, 7.65),
      dashed=True, color=C['arrow_soft'], lw=1.0, rad=0.5)


# ═══════════════════════════════════════════════════════════════════════════
# SEQUENCE-MODEL LEGEND
# ═══════════════════════════════════════════════════════════════════════════
lx, ly = 6.0, 3.95
ax.text(lx, ly, 'Seq. Model:', ha='left', va='center',
        fontsize=8, fontweight='bold', color='#374151', family=FF, zorder=25)

swatches = [
    (lx + 1.65, C['rnn_face'],  C['rnn_edge'],  C['rnn_text'],  'RNN (default)', 1.0),
    (lx + 4.40, C['lstm_face'], C['lstm_edge'], C['lstm_text'], 'LSTM',          0.75),
    (lx + 6.65, C['tf_face'],   C['tf_edge'],   C['tf_text'],   'Transformer',   0.75),
]
for sx, sf, se, st, slabel, salpha in swatches:
    ax.add_patch(FancyBboxPatch(
        (sx, ly - 0.24), 0.42, 0.28,
        boxstyle='round,pad=0.03',
        facecolor=sf, edgecolor=se,
        linewidth=1.4, alpha=salpha, zorder=25,
    ))
    ax.text(sx + 0.55, ly - 0.10, slabel,
            ha='left', va='center', fontsize=8,
            color=st, family=FF, zorder=25)


# ═══════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.3)
plt.savefig('figs/prism_dag_horizontal.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figs/prism_dag_horizontal.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('Saved  figs/prism_dag_horizontal.{png,pdf}')
plt.close()
