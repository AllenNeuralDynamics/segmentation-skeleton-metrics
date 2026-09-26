"""
Created on Fri Sep 26 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for plotting the distributions produced by the skeleton metrics.

"""

import matplotlib.pyplot as plt
import numpy as np
import os


# --- Theme ---
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "text_muted": "#898781",
        "gridline": "#e1e0d9",
        "axis": "#c3c2b7",
        "splits": "#2a78d6",
        "truncations": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "text_muted": "#898781",
        "gridline": "#2c2c2a",
        "axis": "#383835",
        "splits": "#3987e5",
        "truncations": "#d95926",
    },
}

FONT_STACK = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]
MAX_BINS = 40
TAIL_PERCENTILE = 99.0
LOG_Y_RATIO = 50.0


# --- Core Routines ---
def plot_omit_length_distributions(
    split_lengths,
    truncation_lengths,
    output_dir,
    filename="omit_length_distributions.png",
    theme="light",
    dpi=200,
):
    """
    Plots and saves the cable length distributions of omit regions. Splits and
    truncations each get a column with a histogram of counts above an
    empirical CDF, so both the shape and the percentiles are readable.

    Parameters
    ----------
    split_lengths : List[float]
        Cable lengths of omit regions classified as splits.
    truncation_lengths : List[float]
        Cable lengths of omit regions classified as truncations.
    output_dir : str
        Directory where the figure will be saved.
    filename : str, optional
        Name of the saved figure. Default is
        "omit_length_distributions.png".
    theme : str, optional
        Name of a theme in "THEMES", either "light" or "dark". Default is
        "light".
    dpi : int, optional
        Resolution of the saved figure. Default is 200.

    Returns
    -------
    str
        Path to the saved figure.
    """
    palette = THEMES[theme]
    columns = [
        ("Splits", split_lengths, palette["splits"]),
        ("Truncations", truncation_lengths, palette["truncations"]),
    ]
    with plt.rc_context(_rc_params(palette)):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(13, 8.5),
            gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.42,
                         "wspace": 0.18},
        )
        fig.subplots_adjust(top=0.80)
        fig.suptitle(
            "Omit Cable Length Distributions",
            x=0.06,
            y=0.955,
            ha="left",
            fontsize=16,
            fontweight="bold",
            color=palette["text_primary"],
        )
        for j, (name, lengths, color) in enumerate(columns):
            _draw_column(axes[0][j], axes[1][j], name, lengths, color, palette)

        path = os.path.join(output_dir, filename)
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=palette["surface"],
        )
        plt.close(fig)
    return path


def _draw_column(ax_hist, ax_ecdf, name, lengths, color, palette):
    """
    Draws one column of the figure, namely a histogram above an empirical CDF
    for a single collection of cable lengths.

    Parameters
    ----------
    ax_hist : matplotlib.axes.Axes
        Axes that the histogram is drawn on.
    ax_ecdf : matplotlib.axes.Axes
        Axes that the empirical CDF is drawn on.
    name : str
        Name of the omit region type, used as the column heading.
    lengths : List[float]
        Cable lengths to be plotted.
    color : str
        Hex color of the marks in this column.
    palette : dict
        Theme that supplies the surface and text colors.
    """
    values = np.asarray(lengths, dtype=float)
    values = values[np.isfinite(values)]

    # Heading
    _set_heading(ax_hist, name, values, palette)
    if values.size == 0:
        for ax in (ax_hist, ax_ecdf):
            _draw_empty(ax, palette)
        return

    # Marks
    xmax, n_hidden = _view_limit(values)
    _draw_histogram(ax_hist, values, xmax, n_hidden, color, palette)
    _draw_ecdf(ax_ecdf, values, xmax, color, palette)


def _draw_histogram(ax, values, xmax, n_hidden, color, palette):
    """
    Draws a histogram of cable lengths. Adjacent bars are separated by a gap in
    the surface color rather than by a stroke.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes that the histogram is drawn on.
    values : numpy.ndarray
        Cable lengths to be plotted.
    xmax : float
        Upper bound of the plotted range.
    n_hidden : int
        Number of values beyond "xmax", which are noted rather than drawn.
    color : str
        Hex color of the bars.
    palette : dict
        Theme that supplies the surface and text colors.
    """
    # Draw bars
    counts, edges = np.histogram(
        values, bins=_n_bins(values), range=(0.0, xmax)
    )
    ax.bar(
        edges[:-1],
        counts,
        width=np.diff(edges),
        align="edge",
        color=color,
        edgecolor=palette["surface"],
        linewidth=1.0,
        zorder=3,
    )

    # Set scale
    nonzero = counts[counts > 0]
    is_log = nonzero.size > 1 and counts.max() / nonzero.min() > LOG_Y_RATIO
    if is_log:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.7)

    # Style
    _style_axes(ax, palette)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Cable length (μm)", fontsize=10,
                  color=palette["text_secondary"])
    ax.set_ylabel("Count", fontsize=10, color=palette["text_secondary"])
    if n_hidden:
        ax.text(
            0.99,
            0.95,
            f"{n_hidden:,} beyond {_fmt(xmax)} μm not shown",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color=palette["text_muted"],
        )


def _draw_ecdf(ax, values, xmax, color, palette):
    """
    Draws the empirical CDF of the cable lengths, direct labelling only the
    median and the 90th percentile.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes that the empirical CDF is drawn on.
    values : numpy.ndarray
        Cable lengths to be plotted.
    xmax : float
        Upper bound of the plotted range.
    color : str
        Hex color of the curve.
    palette : dict
        Theme that supplies the surface and text colors.
    """
    # Draw curve
    x = np.sort(values)
    y = np.arange(1, x.size + 1) / x.size
    ax.plot(
        np.concatenate(([0.0], x)),
        np.concatenate(([0.0], y)),
        drawstyle="steps-post",
        color=color,
        linewidth=2.0,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )

    # Style
    _style_axes(ax, palette)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel("Cable length (μm)", fontsize=10,
                  color=palette["text_secondary"])
    ax.set_ylabel("Cumulative", fontsize=10,
                  color=palette["text_secondary"])

    # Label percentiles
    for quantile, label in ((0.5, "median"), (0.9, "p90")):
        value = float(np.quantile(values, quantile))
        if value > xmax:
            continue
        ax.plot(
            [value],
            [quantile],
            marker="o",
            markersize=7,
            color=color,
            markeredgecolor=palette["surface"],
            markeredgewidth=2.0,
            zorder=4,
        )
        ax.annotate(
            f"{label} {_fmt(value)} μm",
            xy=(value, quantile),
            xytext=(10, -9),
            textcoords="offset points",
            fontsize=9.5,
            color=palette["text_secondary"],
            va="top",
            zorder=4,
        )


# --- Helpers ---
def _set_heading(ax, name, values, palette):
    """
    Sets the column heading, namely the name of the omit region type above a
    line of summary statistics.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes that the heading is placed above.
    name : str
        Name of the omit region type.
    values : numpy.ndarray
        Cable lengths being summarized.
    palette : dict
        Theme that supplies the text colors.
    """
    ax.set_title(
        name,
        loc="left",
        pad=42,
        fontsize=13,
        fontweight="bold",
        color=palette["text_primary"],
    )
    if values.size > 0:
        n_zero = int(np.count_nonzero(values == 0))
        count = f"n = {values.size:,}"
        if n_zero:
            count += f"   ·   {n_zero:,} zero-length"
        summary = count + (
            f"\nmean {_fmt(values.mean())} ± {_fmt(values.std())} μm   ·   "
            f"median {_fmt(float(np.median(values)))} μm   ·   "
            f"max {_fmt(values.max())} μm"
        )
    else:
        summary = "n = 0"
    ax.text(
        0.0,
        1.02,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        linespacing=1.6,
        color=palette["text_secondary"],
    )


def _style_axes(ax, palette):
    """
    Applies the recessive chrome shared by every panel, namely hairline
    horizontal gridlines, no top or right spine, and muted tick labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to be styled.
    palette : dict
        Theme that supplies the chrome colors.
    """
    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        color=palette["gridline"],
        linewidth=0.8,
        linestyle="-",
        zorder=0,
    )
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(palette["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(
        colors=palette["text_muted"], labelsize=9.5, length=0, pad=6
    )


def _draw_empty(ax, palette):
    """
    Renders a placeholder for an omit region type with no recorded lengths.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to render the placeholder on.
    palette : dict
        Theme that supplies the text colors.
    """
    ax.text(
        0.5,
        0.5,
        "No data",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color=palette["text_muted"],
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines:
        ax.spines[side].set_visible(False)


def _view_limit(values):
    """
    Computes the upper bound of the plotted range by trimming the extreme tail,
    so that a handful of outliers cannot flatten the rest of the distribution.

    Parameters
    ----------
    values : numpy.ndarray
        Cable lengths to be plotted.

    Returns
    -------
    Tuple[float, int]
        Upper bound of the plotted range and the number of values beyond it.
    """
    xmax = float(np.percentile(values, TAIL_PERCENTILE))
    if xmax <= 0:
        xmax = max(float(values.max()), 1.0)
    return xmax, int(np.count_nonzero(values > xmax))


def _n_bins(values):
    """
    Computes the number of histogram bins with the Freedman-Diaconis rule,
    falling back to a smaller count when the data is too sparse for it.

    Parameters
    ----------
    values : numpy.ndarray
        Cable lengths to be plotted.

    Returns
    -------
    int
        Number of histogram bins.
    """
    if values.size < 2:
        return 1

    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    span = float(values.max() - values.min())
    if iqr <= 0 or span <= 0:
        return min(MAX_BINS, max(1, int(np.sqrt(values.size))))

    width = 2 * iqr / np.cbrt(values.size)
    return int(np.clip(np.ceil(span / width), 1, MAX_BINS))


def _fmt(value):
    """
    Formats a cable length for display, dropping the decimal once the value is
    large enough that it carries no information.

    Parameters
    ----------
    value : float
        Cable length to be formatted.

    Returns
    -------
    str
        Formatted cable length.
    """
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:,.1f}"


def _rc_params(palette):
    """
    Builds the matplotlib settings applied while a figure is drawn.

    Parameters
    ----------
    palette : dict
        Theme that supplies the surface and text colors.

    Returns
    -------
    dict
        Settings to be passed to "matplotlib.pyplot.rc_context".
    """
    return {
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "figure.facecolor": palette["surface"],
        "axes.facecolor": palette["surface"],
        "savefig.facecolor": palette["surface"],
        "text.color": palette["text_primary"],
        "axes.labelcolor": palette["text_secondary"],
        "axes.edgecolor": palette["axis"],
        "xtick.color": palette["text_muted"],
        "ytick.color": palette["text_muted"],
    }
