"""Shared figure style for the ICLR paper, matched to ICLR_94/experts_vs_wordlength.pdf.

The reference look: wide-short panel rows, boxed dark spines, dashed grey grid on every
tick, sparse ticks, thick lines with white-edged markers, small framed inset legend.
Colours are the paper's own (tolblue / burgundy) extended with a colourblind-safe
Paul Tol muted set for categorical panels.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, RED = '#4477AA', '#AA3355'                      # paper tolblue + burgundy-leaning red
GREY      = '0.82'
# Paul Tol muted, ordered for the 6 IEMOCAP modalities
CATEGORICAL = ['#4477AA', '#CC6677', '#117733', '#DDCC77', '#88CCEE', '#AA4499']

def apply():
    plt.rcParams.update({
        'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 8.5,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 6.8,
        'axes.linewidth': 1.0, 'axes.edgecolor': '0.25',
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.major.pad': 2, 'ytick.major.pad': 2,
        'lines.linewidth': 1.9, 'lines.markersize': 5.4,
        'lines.markeredgewidth': 1.0, 'lines.markeredgecolor': 'white',
        'legend.framealpha': 0.95, 'legend.borderpad': 0.35,
        'legend.handletextpad': 0.5, 'legend.handlelength': 1.8,
        'legend.labelspacing': 0.25, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

def grid(ax, axis='both'):
    ax.grid(True, axis=axis, ls='--', lw=0.7, color=GREY, zorder=0)
    ax.set_axisbelow(True)

def fade_errorbars(ax, alpha=0.32):
    """Push error bars behind the trend so the lines read first but the spread stays honest."""
    for c in ax.containers:
        if hasattr(c, 'lines') and len(c.lines) > 2 and c.lines[2]:
            for seg in c.lines[2]:
                seg.set_alpha(alpha)
            for cap in c.lines[1]:
                cap.set_alpha(min(1.0, alpha + 0.13))
