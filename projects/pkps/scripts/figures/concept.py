#!/usr/bin/env python
"""Descriptive figure: from unpaired answered-query sets to the product kernel.

Two models i (red) and i' (blue) each answer their OWN subset of a shared query pool.
The PKPS affinity weights every (i-response, i'-response) pair by the query kernel:

    A_{i i'} = sum_{j,l} k_Q(q_j, q_l) k_R(x_ij, x_{i'l}) / sum_{j,l} k_Q(q_j, q_l).

  LEFT  (two stacked views of the SAME queries):
    top    -- queries as dots in embedding space (colour = which model answered; nearby = similar).
    bottom -- the same queries as a row of blocks: model i (red) / model i' (blue) answered subsets.
  CENTER : DKPS, k_Q=delta -> purple only on the diagonal where i and i' answered the SAME
           query (the overlap). Almost everything is discarded.
  RIGHT  : PKPS, k_Q=RBF  -> a dense band; every query pair contributes, weighted by query
           similarity, so responses to *similar* (not identical) queries are used.

red (model i) x blue (model i') -> purple = the product kernel.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

N = 12
RED, BLUE, PURPLE, GRAY = '#3596ff', '#114471', '#6d5bd0', '#cbd5e1'
EMPTY = '#eef2f7'
PCMAP = LinearSegmentedColormap.from_list('pk', ['#ffffff', '#e6e0f8', PURPLE])

# which model answered each of the N queries (unpaired: mostly one model, a few both/neither).
Si = np.array([0, 1, 3, 4, 6, 7, 10])      # model i  (red)
Sip = np.array([1, 2, 4, 5, 7, 8, 9, 11])  # model i' (blue)
inI = np.zeros(N, bool); inI[Si] = True
inP = np.zeros(N, bool); inP[Sip] = True
overlap = np.where(inI & inP)[0]
# per-query label: 0 i-only, 1 i'-only, 2 both, 3 neither
lab = np.where(inI & inP, 2, np.where(inI, 0, np.where(inP, 1, 3)))

# queries live in a 2-D embedding; the kernel matrices below are computed FROM these
# positions over a single UNIVERSAL ordering of the queries, shared by both axes.
rngL = np.random.default_rng(4)
P = np.concatenate([c + rngL.normal(0, 0.3, (4, 2))
                    for c in ([0, 0], [2.7, 0.2], [1.3, 2.4])])     # 3 tight clusters x 4 = 12 queries
t = (P - P.mean(0)) @ np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]   # 1-D order
# order the union of the two models' answered queries by the 1-D embedding projection and
# index BOTH axes by this single ordering. The PKPS query kernel is then a symmetric Gram
# matrix; DKPS's identity kernel is active only on the diagonal, at the shared queries.
# single global ordering by the 1-D projection, shared by the block row in (a) and
# both matrix axes in (b, c): block j corresponds to matrix row/col j
U = np.argsort(t)
sharedU = np.isin(U, overlap)
D2 = ((P[U][:, None] - P[U][None, :]) ** 2).sum(-1)
Krbf = np.exp(-D2 / (2 * 1.7 ** 2))            # PKPS: symmetric query-similarity kernel
Kdkps = np.diag(sharedU.astype(float))         # DKPS: identity, active only at shared queries

# ---- benchmark-score prediction example (right 2x2 block) -------------------
# A small model population with a latent ability that sets each model's true full-benchmark
# score. Each model answers only a few queries from a LARGE shared pool, so two models rarely
# answer the SAME query (DKPS's requirement) although they answer SIMILAR ones (PKPS's). We
# embed the models in each perspective space (classical MDS of the pairwise distances) and
# predict each score by leave-one-out regression onto the embedding. PKPS bridges similar
# queries, recovers ability as a clean gradient, and predicts well; the DKPS limit, restricted
# to the rare shared queries, sees a much noisier geometry and predicts worse.
rngE = np.random.default_rng(0)
NQ, NQA, NM = 160, 9, 16                            # pool size, answered per model, n models
STY, SIG, BR = 1.6, 0.8, 1.8                        # ability signal, per-query noise, k_R bandwidth
qE = np.concatenate([c + rngE.normal(0, 0.35, (NQ // 2, 2)) for c in ([0, 0], [2.6, 1.8])])
a1 = rngE.uniform(-1.5, 1.5, NM)                    # score dimension
a2 = rngE.uniform(-1.0, 1.0, NM) * 0.7             # orthogonal "style" dimension
abil = np.c_[a1, a2]                                # 2-D model ability -> a genuine 2-D perspective
yE = 0.5 + 0.5 * np.tanh(1.2 * a1)                  # true full-benchmark score depends on the score dim
ansE = [np.sort(rngE.choice(NQ, NQA, replace=False)) for _ in range(NM)]
Xe = {(m, q): STY * abil[m] + 0.18 * (qE[q] - qE.mean(0)) + rngE.normal(0, SIG, 2)
      for m in range(NM) for q in ansE[m]}

def _affinity(delta, sig=0.8):                      # bounded RBF k_R (self-affinity carries no ability leak)
    A = np.zeros((NM, NM))
    for a in range(NM):
        for b in range(NM):
            num = den = 0.0
            for qa in ansE[a]:
                for qb in ansE[b]:
                    kq = (1.0 if qa == qb else 0.0) if delta else \
                        np.exp(-((qE[qa] - qE[qb]) ** 2).sum() / (2 * sig ** 2))
                    if kq:
                        kr = np.exp(-((Xe[(a, qa)] - Xe[(b, qb)]) ** 2).sum() / (2 * BR ** 2))
                        num += kq * kr; den += kq
            A[a, b] = num / den if den > 0 else 0.0
    return A

def _mds2(A):                                       # classical MDS to 2-D from an affinity
    d2 = np.maximum(np.diag(A)[:, None] + np.diag(A)[None, :] - 2 * A, 0)
    J = np.eye(NM) - 1.0 / NM
    w, V = np.linalg.eigh(-0.5 * J @ d2 @ J)
    idx = np.argsort(w)[::-1][:2]
    return V[:, idx] * np.sqrt(np.maximum(w[idx], 0))

def _align(Z):       # MDS orientation is arbitrary; rotate so the score gradient runs along +x
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(NM), Z], yE, rcond=None)
    d = beta[1:]; d = d / (np.linalg.norm(d) + 1e-12)
    return Z @ np.array([[d[0], d[1]], [-d[1], d[0]]]).T

def _loo_predict(Z):                                # predict each score from the others' embeddings
    Zb = np.c_[np.ones(NM), Z]
    pred = np.zeros(NM)
    for m in range(NM):
        tr = np.arange(NM) != m
        beta, *_ = np.linalg.lstsq(Zb[tr], yE[tr], rcond=None)
        pred[m] = np.clip(Zb[m] @ beta, 0, 1)
    return pred

Zdk_raw, Zpk_raw = _mds2(_affinity(True)), _mds2(_affinity(False))
pred_dk, pred_pk = _loo_predict(Zdk_raw), _loo_predict(Zpk_raw)
Zdk, Zpk = _align(Zdk_raw), _align(Zpk_raw)         # rotation is display-only (LOO is rotation-invariant)
mae_dk, mae_pk = np.mean(np.abs(pred_dk - yE)), np.mean(np.abs(pred_pk - yE))

fig = plt.figure(figsize=(11.5, 3.3))   # aspect preserved (15.5:4.7) so the fig-fraction layout
                                         # below is unchanged; narrower fig -> fonts render larger
                                         # at \textwidth (squeeze 0.48, matching the other figures)
gs = GridSpec(2, 5, width_ratios=[0.82, 1, 1, 0.5, 0.5], height_ratios=[0.58, 1.42],
              wspace=0.22, hspace=0.05, top=0.92, bottom=0.10)
axBlk = fig.add_subplot(gs[0, 0])
axEmb = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[:, 1])
axR = fig.add_subplot(gs[:, 2])
axDKe = fig.add_subplot(gs[0, 3]); axDKp = fig.add_subplot(gs[0, 4])     # DKPS embed | pred
axPKe = fig.add_subplot(gs[1, 3]); axPKp = fig.add_subplot(gs[1, 4])     # PKPS embed | pred

# ---- LEFT TOP: queries as dots in embedding space ----------------------------
for k, col, sz, z in [(3, GRAY, 42, 1), (0, RED, 105, 3), (1, BLUE, 105, 3), (2, PURPLE, 125, 4)]:
    m = lab == k
    axEmb.scatter(P[m, 0], P[m, 1], s=sz, c=col, edgecolors='white', lw=1.2, zorder=z)
axEmb.set_aspect('equal'); axEmb.axis('off')
axEmb.set_title('query embedding\n(nearby = similar)', fontsize=10.5, pad=6, linespacing=0.95)

# ---- LEFT BOTTOM: same queries as a row of answered blocks --------------------
for pos, j in enumerate(U):        # same similarity ordering as the matrix axes
    axBlk.add_patch(Rectangle((pos, 1), 1, 1, facecolor=RED if inI[j] else EMPTY,
                              edgecolor='white', lw=1.5))
    axBlk.add_patch(Rectangle((pos, 0), 1, 1, facecolor=BLUE if inP[j] else EMPTY,
                              edgecolor='white', lw=1.5))
axBlk.set_xlim(-0.3, N); axBlk.set_ylim(-0.4, 2.4)
axBlk.set_aspect('equal'); axBlk.set_anchor('C'); axBlk.axis('off')
axBlk.text(-0.45, 1.5, 'model $i$', ha='right', va='center', fontsize=11, color=RED)
axBlk.text(-0.45, 0.5, "model $i'$", ha='right', va='center', fontsize=11, color=BLUE)
# ---- CENTER & RIGHT: the kernel-weight matrices ------------------------------
for ax, W in [(axC, Kdkps), (axR, Krbf)]:
    im = ax.imshow(W, cmap=PCMAP, vmin=0, vmax=1, origin='upper', aspect='equal')
    ax.set_xticks(np.arange(-.5, W.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, W.shape[0], 1), minor=True)
    ax.grid(which='minor', color='#eef1f6', lw=0.5)
    ax.tick_params(which='minor', length=0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor('#d7dde6')
    ax.set_xlabel('query index', fontsize=10, color='#486884')
axC.set_ylabel('query index', fontsize=10, color='#486884')

# ---- RIGHT 2x2 BLOCK: perspective embedding (coloured by score) + LOO prediction
SCM = 'viridis'
for axe, axp, Z, pred, mae, le, lp in [(axDKe, axDKp, Zdk, pred_dk, mae_dk, 'd', "d'"),
                                       (axPKe, axPKp, Zpk, pred_pk, mae_pk, 'e', "e'")]:
    # MDS 2-D model representations (one dot per model), coloured by true benchmark score;
    # the embedding is rotated so the score gradient runs left->right
    sc = axe.scatter(Z[:, 0], Z[:, 1], c=yE, cmap=SCM, vmin=0, vmax=1,
                     s=54, edgecolors='white', lw=0.8, zorder=3)
    # fixed square frame (equal x/y range, with padding) so the box never auto-resizes and
    # all four panels stay aligned; preserves the true embedding geometry
    cx, cy = Z[:, 0].mean(), Z[:, 1].mean()
    rr = 0.66 * max(np.ptp(Z[:, 0]), np.ptp(Z[:, 1]))
    axe.set_xlim(cx - rr, cx + rr); axe.set_ylim(cy - rr, cy + rr)
    axe.set_aspect('equal'); axe.set_xticks([]); axe.set_yticks([])
    axe.text(0.05, 0.95, f'({le})', transform=axe.transAxes, ha='left', va='top',
             fontsize=9, fontweight='bold', color='#213c66', zorder=4,
             bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.65))
    for s in axe.spines.values():
        s.set_edgecolor('#d7dde6')
    # predict each model's score by LOO regression onto its 2-D representation
    axp.plot([0, 1], [0, 1], ls='--', color='#94a3b8', lw=1.2, zorder=1)
    axp.scatter(yE, pred, c=yE, cmap=SCM, vmin=0, vmax=1, s=44, edgecolors='white', lw=0.8, zorder=3)
    axp.set_xlim(-0.05, 1.05); axp.set_ylim(-0.05, 1.05); axp.set_aspect('equal')
    axp.set_xticks([0, 1]); axp.set_yticks([0, 1]); axp.tick_params(labelsize=8, color='#d7dde6')
    axp.text(0.05, 0.94, f'({lp})', transform=axp.transAxes, ha='left', va='top',
             fontsize=9, fontweight='bold', color='#213c66', zorder=4,
             bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.65))
    axp.text(0.95, 0.07, f'MAE {mae:.2f}', transform=axp.transAxes, ha='right', va='bottom',
             fontsize=8.5, color='#213c66')
    for s in axp.spines.values():
        s.set_edgecolor('#d7dde6')
axPKp.set_xlabel('true score', fontsize=8.5, color='#486884', labelpad=1)

# (the matrix colour key is added after layout, once (c) is moved next to (b))

# place the (a)/(b)/(c) panel titles at a common figure-y. The panels have different box
# tops (the matrices are letterbox-centered), so per-axes titles can't line up -- figure-
# coordinate text over each panel centre does.
fig.canvas.draw()
# pull the answered-query blocks down so their top aligns with the matrices' top: this
# tightens the tall gap to the query embedding below and lets the (a)/(b)/(c) titles drop
mt = axC.get_position().y1
pb = axBlk.get_position()
axBlk.set_position([pb.x0, mt - pb.height, pb.width, pb.height])
# bring the PKPS matrix (c) tight next to the DKPS matrix (b) for side-by-side comparison,
# then place the query-kernel colour key just to (c)'s right
pc = axC.get_position()
axR.set_position([pc.x1 + 0.022, pc.y0, pc.width, pc.height])
mcax = fig.add_axes([axR.get_position().x1 + 0.012, pc.y0, 0.008, pc.height])
mcb = fig.colorbar(im, cax=mcax)
mcax.set_title(r'$k_Q$', fontsize=9, pad=3, color='#486884')   # label on top: the flow arrow passes mid-height
mcb.set_ticks([0, 1])
# arrange the right 2x2 block as flush PHYSICAL squares spanning the matrices' height,
# packed tight to save space
pr = axR.get_position()
gap_v = 0.09
h = (pr.height - gap_v) / 2
side = h * (fig.get_figheight() / fig.get_figwidth())   # fig-fraction width of a square of height h
x0 = pr.x1 + 0.058                                       # clear the (now-compact) matrix colour key
hgap = 0.014
for axe, axp, ytop in [(axDKe, axDKp, pr.y1), (axPKe, axPKp, pr.y0 + h)]:
    axe.set_position([x0, ytop - h, side, h])
    axp.set_position([x0 + side + hgap, ytop - h, side, h])

# slim colour key (true benchmark score) for the embedding + prediction dots
scax = fig.add_axes([x0 + 2 * side + hgap + 0.012, pr.y0, 0.006, pr.height])
scb = fig.colorbar(sc, cax=scax); scb.set_label('true score', fontsize=8.5, labelpad=-2)
scb.set_ticks([0, 1]); scb.ax.tick_params(labelsize=7)

Y = max(ax.get_position().y1 for ax in (axBlk, axC, axR)) + 0.012
for ax, t in [(axBlk, r'$\bf{(a)}$  answered queries'),
              (axC, r'$\bf{(b)}$  DKPS  ·  $k_Q=\delta$'),
              (axR, r'$\bf{(c)}$  PKPS  ·  $k_Q=$ RBF')]:
    p = ax.get_position()
    fig.text((p.x0 + p.x1) / 2, Y, t, ha='center', va='bottom', fontsize=12)
# block column headers (embedding | prediction) just above the top row, and DKPS/PKPS row
# labels in brand colours
pe2, pp2 = axDKe.get_position(), axDKp.get_position()
hY = pe2.y1 + 0.016
fig.text((pe2.x0 + pe2.x1) / 2, hY, 'model\nembedding', ha='center', va='bottom',
         fontsize=9.5, linespacing=0.95)
fig.text((pp2.x0 + pp2.x1) / 2, hY, 'predicted vs true\nbenchmark score', ha='center',
         va='bottom', fontsize=9.5, linespacing=0.95)
for ax, lab, col in [(axDKe, 'DKPS', '#93aacc'), (axPKe, 'PKPS', '#3596ff')]:
    p = ax.get_position()
    fig.text(p.x0 - 0.009, (p.y0 + p.y1) / 2, lab, ha='right', va='center', fontsize=11,
             fontweight='bold', color=col, rotation=90)

# size the embedding to fill the lower-left region, and put a shared legend at the bottom
# (fixed in figure coords) so the left column bottoms out level with the matrices' labels.
pe = axEmb.get_position(); pm = axC.get_position()
axEmb.set_position([pe.x0, 0.235, pe.width, pe.height * 0.74])
xc = (axBlk.get_position().x0 + axBlk.get_position().x1) / 2
fig.legend(handles=[Line2D([0], [0], marker='s', ls='', mfc=RED, mec='white', ms=9, label='model $i$'),
                    Line2D([0], [0], marker='s', ls='', mfc=BLUE, mec='white', ms=9, label="model $i'$"),
                    Line2D([0], [0], marker='o', ls='', mfc=PURPLE, mec='white', ms=9, label='both')],
           loc='lower center', bbox_to_anchor=(xc, 0.125), bbox_transform=fig.transFigure,
           ncol=3, frameon=False, fontsize=9.5, handletextpad=0.2, columnspacing=1.0)

for ext in ('png', 'pdf'):
    fig.savefig(f'results-pkps-rd1/fig_concept.{ext}', dpi=200, bbox_inches='tight')
print('wrote results-pkps-rd1/fig_concept.png')
