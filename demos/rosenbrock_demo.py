
from numpy import array, where, zeros, exp
from imageio import mimwrite, imread
from os import remove
from itertools import chain
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from pdfgrid import PdfGrid


"""
This script builds a .gif file of PdfGrid evaluating the rosenbrock density
for diagnostic purposes
"""


def rosenbrock(t):
    # This is a modified form of the rosenbrock function, which
    # is commonly used to test optimisation algorithms
    X, Y = t
    Y += 1
    X2 = X**2
    b = 15  # correlation strength parameter
    v = 3   # variance of the gaussian term
    return -X2 - b*(Y - X2)**2 - 0.5*(X2 + Y**2)/v

grid_spacing = array([0.1, 0.1])
grid_centre = array([0., 0.])
SPG = PdfGrid(spacing=grid_spacing, offset=grid_centre)


image_id = 0
files = []
while SPG.state != "end":
    P = array([rosenbrock(theta) for theta in SPG.get_parameters()])
    SPG.give_probabilities(P)


    grid = zeros([60, 60]) + 2
    for v,p in zip(SPG.coordinates, SPG.probability):
        i, j = v
        grid[i+30,j+30] = p

    inds = where(grid == 2)
    grid = exp(grid)
    grid[inds] = -1

    current_cmap = get_cmap()
    current_cmap.set_under('white')

    filename = f'rosenbrock_{image_id}.png'
    files.append(filename)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111)
    ax.imshow(grid.T, interpolation='nearest', vmin=0., vmax=1.)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(filename, dpi = 50)
    plt.close()

    image_id += 1

images = []
for filename in chain(files, [files[-1]]*20):
    images.append(imread(filename))

mimwrite('PdfGrid.gif', images, duration = 0.05)

for filename in files:
    remove(filename)