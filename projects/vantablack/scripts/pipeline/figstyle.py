"""Text sizes for paper figures, set from their size on the printed page.

A figure drawn `width_in` inches wide and included at the page width `page_in` (ICML two-column
\\textwidth, 6.75 in) is scaled by page_in / width_in. apply_print_sizes sets the shared style's
sizes so that, after that scaling, panel titles, axis labels and legends, and tick labels print
at the given point sizes. Call it after hv.apply() and before drawing.
"""
import matplotlib as mpl

PAGE_IN = 6.75


def apply_print_sizes(hv, width_in, page_in=PAGE_IN, title=8.5, label=7.5, tick=6.5):
    k = width_in / page_in
    hv.SIZES.update(subtitle=title * k, label=label * k, annot=label * k, legend=label * k, tick=tick * k)
    mpl.rcParams.update({'font.size': label * k, 'xtick.labelsize': tick * k, 'ytick.labelsize': tick * k,
                         'legend.fontsize': label * k, 'legend.title_fontsize': label * k})
    return k


def below_axes(fig, pad=0.012):
    """Figure y (0-1) just below the lowest drawn element of any axes (tick labels and axis
    labels included), for placing a legend under the panels without a gap."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    return min(inv.transform(ax.get_tightbbox(rend).corners())[:, 1].min() for ax in fig.axes) - pad
