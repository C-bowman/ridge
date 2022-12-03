
from imageio.v2 import mimwrite, imread
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from numpy import array, where, zeros, exp
from os import remove
from itertools import chain

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


def main() -> None:
    # Set up adaptive grid sampler
    grid_spacing = array([0.1,0.1])
    SPG = PdfGrid(2)
    # Run adaptive grid sampling
    image_id = 0
    files = []
    while SPG.state != "end":
        P = [rosenbrock(theta*grid_spacing) for theta in SPG.to_evaluate]
        SPG.update_cells(P)
        SPG.take_step()
        # Create image showing the progress in mapping the posterior
        grid = zeros([60, 60]) + 2
        for v,p in zip(SPG.indices, SPG.probability):
            i,j = v
            grid[i+30,j+30] = p

        inds = where(grid == 2)
        grid = exp(grid)
        grid[inds] = -1

        current_cmap = get_cmap()
        current_cmap.set_under('white')

        filename = 'rosenbrock_{}.png'.format(image_id)
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
    # Build GIF to show the evolution of the adaptive grid sampler
    images = []
    for filename in chain(files, [files[-1]]*20):
        images.append(imread(filename))
    # Save the GIF
    mimwrite('PdfGrid.gif', images, duration = 0.05)
    # Delete the composit images
    for filename in files:
        remove(filename)


if __name__ == "__main__":
    main()
