import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'matplotlib'))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({                              # clean modern style: white, subtle grid, despined
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#afbec6', 'axes.linewidth': 0.9,
    'axes.grid': True, 'axes.axisbelow': True, 'grid.color': '#e5e7eb', 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': '#486884', 'ytick.color': '#486884',
    'axes.labelcolor': '#213c66', 'axes.titlecolor': '#0a2245', 'text.color': '#213c66', 'font.size': 11,
})
import numpy as np
import pandas as pd


# the full PKPS (combined, uses all data) is red like PKPS elsewhere; the paired/unpaired
# ablations get their own colours (teal/orange) so they never collide with a method's colour
# (blue = matrix completion, green = DKPS, purple = IRT in the other figures).
ESTIMATOR_COLORS = {'sample': '#9ca3af', 'dkps': '#93aacc', 'pkps': '#3596ff'}
ESTIMATOR_LABELS = {'sample': 'Sample score', 'dkps': 'DKPS', 'pkps': 'PKPS'}
# HELM figure convention: score-only methods dashed, embedding methods solid
ESTIMATOR_LS = {'sample': '--', 'dkps': '-', 'pkps': '-'}
ESTIMATOR_ORDER = ['sample', 'dkps', 'pkps']


def _aggregate(df, group_cols, value_col):
    grouped = (
        df.groupby(group_cols, dropna=False)[value_col]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    grouped['sem'] = grouped['std'].fillna(0.0) / np.sqrt(grouped['count'].clip(lower=1))
    return grouped


def _errbar(ax, x, mean, sem, color, label, ls='-'):
    ax.errorbar(x, mean, yerr=sem, marker='o', ms=4, lw=2.6, color=color, ls=ls,
                label=label, capsize=2, elinewidth=0.8)


def _plot_standard_panel(ax, df, x_col, xlabel):
    """DKPS vs PKPS, x vs MAE (no per-panel legend; the figure has one shared legend)."""
    summary = _aggregate(df, [x_col, 'estimator'], 'mae')
    for est in ESTIMATOR_ORDER:
        if est not in summary['estimator'].values:
            continue
        curve = summary[summary['estimator'] == est].sort_values(x_col)
        _errbar(ax, curve[x_col], curve['mean'], curve['sem'],
                ESTIMATOR_COLORS[est], ESTIMATOR_LABELS[est], ls=ESTIMATOR_LS[est])
    ax.set_xlabel(xlabel)


def _plot_rho(ax, df, xlabel):
    """Cross-model overlap rho: DKPS (delta) collapses as rho->0, PKPS (rbf) holds."""
    summary = _aggregate(df, ['rho', 'estimator'], 'mae')
    for est in ESTIMATOR_ORDER:
        if est not in summary['estimator'].values:
            continue
        curve = summary[summary['estimator'] == est].sort_values('rho')
        _errbar(ax, curve['rho'], curve['mean'], curve['sem'],
                ESTIMATOR_COLORS[est], ESTIMATOR_LABELS[est], ls=ESTIMATOR_LS[est])
    ax.set_xlabel(xlabel)


def _plot_query_efficiency(ax, df, xlabel):
    """MAE vs per-cell query budget at the unpaired design (rho=0), matching the other
    panels; the pairing dimension itself is the rho-sweep panel."""
    df = df[(df['rho'] == 0) | (df['estimator'] == 'sample')]
    summary = _aggregate(df, ['budget', 'estimator'], 'mae')
    for est in ESTIMATOR_ORDER:
        curve = summary[summary['estimator'] == est].sort_values('budget')
        _errbar(ax, curve['budget'], curve['mean'], curve['sem'],
                ESTIMATOR_COLORS[est], None, ls=ESTIMATOR_LS[est])
    ax.set_xlabel(xlabel)
    ax.set_xscale('log', base=2)


# Panel specs: (experiment, x_col, xlabel, custom_fn, fixed-parameter title)
# (a)-(e) run the query-efficiency (denoising) protocol: every observed cell has a
# budget of M queries and we predict its TRUE score; the gray line is the cell's own
# M-query sample score. (f) runs the completion protocol (held-out cells; gray =
# task-mean of observed scores).
PANELS = [
    ('query_efficiency', None, r'queries per cell $m$', _plot_query_efficiency,
     r'$n{=}60,\ T{=}12,\ \rho{=}0,\ p_{\mathrm{task}}{=}1$'),
    ('qe_rho', None, r"query overlap $\rho=m_{ii'}/m$", _plot_rho,
     r'$n{=}60,\ T{=}12,\ m{=}4,\ p_{\mathrm{task}}{=}1$'),
    ('qe_n_models', 'n_models', r'number of models $n$', None,
     r'$T{=}12,\ m{=}4,\ \rho{=}0,\ p_{\mathrm{task}}{=}1$'),
    ('qe_n_tasks', 'n_tasks', r'number of tasks $T$', None,
     r'$n{=}60,\ m{=}4,\ \rho{=}0,\ p_{\mathrm{task}}{=}1$'),
    ('qe_task_parity', 'obs_prob', r'task coverage $p_{\mathrm{task}}$', None,
     r'$n{=}60,\ T{=}12,\ m{=}4,\ \rho{=}0$'),
    ('task_parity', 'obs_prob', r'task coverage $p_{\mathrm{task}}$ (completion)', None,
     r'$n{=}100,\ T{=}20,\ M{=}10,\ p_{\mathrm{query}}{=}0.8$'),
]


def plot_figure(results, nrows=2, ncols=3, figsize=(11.5, 4.9)):
    """Create the main 2x3 figure with one shared legend and per-panel fixed-parameter titles."""
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharey=True)

    for idx, (exp_name, x_col, xlabel, custom_fn, fixed) in enumerate(PANELS):
        ax = axes[idx // ncols][idx % ncols]
        if exp_name not in results or results[exp_name] is None:
            ax.set_visible(False)
            continue
        df = results[exp_name]
        if custom_fn is not None:
            custom_fn(ax, df, xlabel)
        else:
            _plot_standard_panel(ax, df, x_col, xlabel)
        if idx % ncols == 0:
            ax.set_ylabel(r'MAE$(\hat{y}, y)$', fontsize=15.6)
        ax.set_xlabel(ax.get_xlabel(), fontsize=15.6)
        ax.tick_params(labelsize=13.1)
        # HELM figure conventions: bold corner letter as the left title; fixed
        # parameters right-aligned so the long strings never collide with it
        ax.set_title(f'({chr(97 + idx)})', loc='left', fontweight='bold', fontsize=16.2)
        ax.set_title(fixed, loc='right', fontsize=13.8)

    axes[0][0].set_ylim(top=2.5)  # shared y; cap so panel (f)'s high-noise tail doesn't stretch all panels
    handles = [Line2D([0], [0], color=ESTIMATOR_COLORS[e], marker='o', lw=2.6,
                      ls=ESTIMATOR_LS[e], label=ESTIMATOR_LABELS[e]) for e in ESTIMATOR_ORDER]
    fig.tight_layout()
    # legend flush above the top row's panel titles, suptitle flush above the legend,
    # both centred on the panel span (canvas centre is skewed left by the shared ylabel)
    fig.canvas.draw()
    r, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    flat = [ax for row in axes for ax in row if ax.get_visible()]
    xc = 0.5 * (min(ax.get_position().x0 for ax in flat) +
                max(ax.get_position().x1 for ax in flat))
    ytop = max(ax.get_tightbbox(r).transformed(inv).y1 for ax in flat)
    lg = fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False,
                    bbox_to_anchor=(xc, ytop + 0.010), fontsize=13.8, handlelength=3.2)
    fig.canvas.draw()
    fig.suptitle('Synthetic study', fontsize=17.5, fontweight='bold',
                 x=xc, y=lg.get_window_extent(r).transformed(inv).y1 + 0.012, va='bottom')
    # shrink any params title that still collides with its corner letter
    for ax in flat:
        params = ax.get_title(loc='right')
        if not params:
            continue
        size = 13.8
        while size > 10.5:
            fig.canvas.draw()
            rr = fig.canvas.get_renderer()
            if ax._left_title.get_window_extent(rr).x1 + 4 < \
                    ax._right_title.get_window_extent(rr).x0:
                break
            size -= 0.7
            ax.set_title(params, loc='right', fontsize=size)
    return fig


def save_figure(output_dir, results, filename='fig_synthetic'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_figure(results)
    for ext in ('pdf', 'png'):
        fig.savefig(output_dir / f'{filename}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
