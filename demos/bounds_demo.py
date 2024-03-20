from numpy import sqrt, array
import matplotlib.pyplot as plt
from pdfgrid import PdfGrid
from pdfgrid.plotting import plot_marginal_2d

"""
2D test case using a ring-shaped distribution centred on the origin, with
bounds on the parameters deliberately set to exclude a portion of the ring.
"""


class RingGaussian:
    def __init__(self):
        self.R0 = 1.
        self.eps = 0.10
        self.coeff = -0.5 / (self.R0*self.eps)**2

    def __call__(self, theta):
        x, y = theta
        r_sqr = (sqrt(x**2 + y**2) - self.R0)**2
        return self.coeff * r_sqr


# create an instance of the posterior
posterior = RingGaussian()

# specify settings for the grid
grid_spacing = array([0.03, 0.03])
grid_centre = array([0., 0.])
grid_bounds = array([[-0.5, 0.], [1.5, 1.5]]).T

grid = PdfGrid(
    spacing=grid_spacing,
    offset=grid_centre,
    bounds=grid_bounds
)

# evaluate the posterior
grid.evaluate_posterior(posterior=posterior)

# evaluate and plot the 1D marginal distribution for the first parameter
points, probs = grid.get_marginal([0])
plt.plot(points, probs, lw=2)
plt.fill_between(points, 0., probs, color="C0", alpha=0.1)
plt.grid()
plt.ylim([0, None])
plt.xlabel("x")
plt.ylabel("probability density")
plt.tight_layout()
plt.show()

# plot the full 2D distribution
points, probs = grid.get_marginal([0, 1])
plot_marginal_2d(points=points, probabilities=probs, labels=["x", "y"])

# plot the convergence information
grid.plot_convergence()
