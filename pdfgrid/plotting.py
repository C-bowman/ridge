import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib import colormaps
from numpy import array, diff, sort, ndarray


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


def plot_marginal_2d(points, probabilities):

    spacing = array([find_spacing(v) for v in points.T])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    rectangles = [Rectangle(v, *spacing) for v in points - 0.5 * spacing[None, :]]

    x_limits = [points[:, 0].min(), points[:, 0].max()]
    y_limits = [points[:, 1].min(), points[:, 1].max()]

    # get a color for each of the rectangles
    colormap = "viridis"
    cmap = colormaps.get_cmap(colormap)
    rectangle_colors = cmap(probabilities / probabilities.max())

    pc = PatchCollection(
        rectangles, facecolors = rectangle_colors
    )

    ax.add_collection(pc)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    plt.tight_layout()
    plt.show()


def find_spacing(values: ndarray):
    diffs = diff(sort(values))
    return diffs.compress(diffs > 0.).min()
