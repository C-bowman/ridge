from collections.abc import Sequence
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib import colormaps
from numpy import array, diff, sort, ndarray
from pdfgrid.utils import compute_marginal


def plot_convergence(evaluations, probabilities):
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121)
    ax1.plot(evaluations, probabilities, ".-", lw=2)
    ax1.set_xlabel("total posterior evaluations")
    ax1.set_ylabel("total probability of evaluated cells")
    ax1.grid()

    ax2 = fig.add_subplot(122)
    p = array(probabilities[1:])
    frac_diff = p[1:] / p[:-1] - 1
    ax2.plot(evaluations[2:], frac_diff, alpha=0.5, lw=2, c="C0")
    ax2.plot(evaluations[2:], frac_diff, "D", c="C0")
    ax2.set_xlim([0.0, None])
    ax2.set_yscale("log")
    ax2.set_xlabel("total posterior evaluations")
    ax2.set_ylabel("fractional change in total probability")
    ax2.grid()

    plt.tight_layout()
    plt.show()


def plot_marginal_2d(
    points: ndarray,
    probabilities: ndarray,
    labels: list[str] = None,
    plot_axis=None,
    colormap_name: str = "viridis"
):
    spacing = array([find_spacing(v) for v in points.T])

    if plot_axis is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = plot_axis

    rectangles = [Rectangle(v, *spacing) for v in points - 0.5 * spacing[None, :]]

    x_limits = [points[:, 0].min(), points[:, 0].max()]
    y_limits = [points[:, 1].min(), points[:, 1].max()]

    # get a color for each of the rectangles
    cmap = colormaps[colormap_name]
    rectangle_colors = cmap(probabilities / probabilities.max())

    pc = PatchCollection(rectangles, facecolors=rectangle_colors)

    ax.add_collection(pc)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

    if isinstance(labels, Sequence) and len(labels) >= 2:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

    if plot_axis is None:
        plt.tight_layout()
        plt.show()


def find_spacing(values: ndarray):
    diffs = diff(sort(values))
    return diffs.compress(diffs > 0.).min()


def matrix_plot(
    coords: ndarray,
    probs: ndarray,
    spacing: ndarray,
    offset: ndarray,
    labels=None,
    show=True,
    reference=None,
    filename=None,
    colormap="viridis",
    show_ticks=None,
    label_size=10,
):
    """
    Construct a 'matrix plot' of the parameters which shows all possible
    1D and 2D marginal distributions.

    :param coords: \
        Grid-point coordinates of the evaluated cells as a numpy ``ndarray``.

    :param probs: \
        Probabilities (up to a multiplicative constant) of the evaluated cells
        as a numpy ``ndarray``.

    :param spacing: \
        The grid spacing in each dimension as a numpy ``ndarray``.

    :param offset: \
        The grid offset (i.e. the parameter values corresponding the grid origin)
        as a numpy ``ndarray``.

    :keyword labels: \
        A list of strings to be used as axis labels for each parameter being plotted.

    :keyword bool show: \
        Sets whether the plot is displayed.

    :keyword reference: \
        A list of reference values for each parameter which will be over-plotted.

    :keyword str filename: \
        File path to which the matrix plot will be saved (if specified).

    :keyword str colormap: \
        Name of a ``matplotlib`` colormap to be used for the plots.

    :keyword bool show_ticks: \
        By default, axis ticks are only shown when plotting less than 6 variables.
        This behaviour can be overridden for any number of parameters by setting
        show_ticks to either True or False.

    :keyword int label_size: \
        The font-size used for axis labels.
    """

    N_par = spacing.size
    if labels is None:  # set default axis labels if none are given
        if N_par >= 10:
            labels = [f"p{i}" for i in range(N_par)]
        else:
            labels = [f"param {i}" for i in range(N_par)]
    else:
        if len(labels) != N_par:
            raise ValueError(
                """
                [ matrix_plot error ]
                >> The number of labels given does not match
                >> the number of plotted parameters.
                """
            )

    if reference is not None:
        if len(reference) != N_par:
            raise ValueError(
                """
                [ matrix_plot error ]
                >> The number of reference values given does not match
                >> the number of plotted parameters.
                """
            )

    # by default, we suppress axis ticks if there are 6 parameters or more to keep things tidy
    if show_ticks is None:
        show_ticks = N_par < 6

    cmap = colormaps[colormap]
    # find the darker of the two ends of the colormap, and use it for the marginal plots
    marginal_color = sorted([cmap(10), cmap(245)], key=lambda x: sum(x[:-1]))[0]

    fig = plt.figure(figsize=(8, 8))
    # build a lower-triangular indices list in diagonal-striped order
    inds_list = [(N_par - 1, 0)]  # start with bottom-left corner
    for k in range(1, N_par):
        inds_list.extend([(N_par - 1 - i, k - i) for i in range(k + 1)])

    # now create a dictionary of axis objects with correct sharing
    axes = {}
    for tup in inds_list:
        i, j = tup
        x_share = None
        y_share = None

        if i < N_par - 1:
            x_share = axes[(N_par - 1, j)]

        if (j > 0) and (i != j):  # diagonal doesnt share y-axis
            y_share = axes[(i, 0)]

        axes[tup] = plt.subplot2grid(
            (N_par, N_par), (i, j), sharex=x_share, sharey=y_share
        )

    # now loop over grid and plot
    for tup in inds_list:
        i, j = tup
        ax = axes[tup]
        # are we on the diagonal?
        if i == j:
            points, marginal_probs = compute_marginal(
                coords=coords,
                probs=probs,
                spacing=spacing,
                offset=offset,
                z=[i]
            )

            ax.plot(
                points,
                0.9 * (marginal_probs / marginal_probs.max()),
                lw=1,
                color=marginal_color,
            )
            ax.fill_between(
                points,
                0.9 * (marginal_probs / marginal_probs.max()),
                color=marginal_color,
                alpha=0.1,
            )

            if reference is not None:
                ax.plot(
                    [reference[i], reference[i]],
                    [0, 1],
                    lw=1.5,
                    ls="dashed",
                    color="red",
                )
            ax.set_ylim([0, 1])

        else:
            points, marginal_probs = compute_marginal(
                coords=coords,
                probs=probs,
                spacing=spacing,
                offset=offset,
                z=[j, i],
            )

            plot_marginal_2d(
                points=points,
                probabilities=marginal_probs,
                plot_axis=ax,
                colormap_name=colormap
            )

            # plot any reference points if given
            if reference is not None:
                ax.plot(
                    reference[j],
                    reference[i],
                    marker="o",
                    markersize=7,
                    markerfacecolor="none",
                    markeredgecolor="white",
                    markeredgewidth=3.5,
                )
                ax.plot(
                    reference[j],
                    reference[i],
                    marker="o",
                    markersize=7,
                    markerfacecolor="none",
                    markeredgecolor="red",
                    markeredgewidth=2,
                )

        # assign axis labels
        if i == N_par - 1:
            ax.set_xlabel(labels[j], fontsize=label_size)
        if j == 0 and i != 0:
            ax.set_ylabel(labels[i], fontsize=label_size)

        if show_ticks:  # set up ticks for the edge plots if they are to be shown
            # hide x-tick labels for plots not on the bottom row
            if i < N_par - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            # hide y-tick labels for plots not in the left column
            if j > 0:
                plt.setp(ax.get_yticklabels(), visible=False)
            # remove all y-ticks for 1D marginal plots on the diagonal
            if i == j:
                ax.set_yticks([])
        else:  # else remove all ticks from all axes
            ax.set_xticks([])
            ax.set_yticks([])

    # set the plot spacing
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    # save/show the figure if required
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()

    return fig