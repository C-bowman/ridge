from numpy import sqrt, array
import matplotlib.pyplot as plt
from pdfgrid import PdfGrid
from pdfgrid.plotting import plot_marginal_2d


class ToroidalGaussian:
    def __init__(self):
        self.R0 = 1.
        self.eps = 0.10
        self.coeff = -0.5 / (self.R0*self.eps)**2

    def __call__(self, theta):
        x, y, z = theta
        r_sqr = z**2 + (sqrt(x**2 + y**2) - self.R0)**2
        return self.coeff * r_sqr


# set up the test-case posterior
posterior = ToroidalGaussian()

# specify settings for the grid
grid_spacing = array([0.04, 0.04, 0.02])
grid_centre = array([0., 0., 0.])
grid_bounds = array([[-1.5, -1.5, -0.5], [1.5, 1.5, 0.5]]).T

# create a PdfGrid instance
grid = PdfGrid(
    spacing=grid_spacing,
    offset=grid_centre,
    bounds=grid_bounds
)

while grid.state != "end":
    # get the next batch of parameter evaluations
    params = grid.get_parameters()
    # evaluate the posterior log-probabilities
    P = array([posterior(theta) for theta in params])
    # pass the log-probabilities back to PdfGrid
    grid.give_probabilities(P)


# evaluate and plot the marginal for the first dimension
points, probs = grid.get_marginal([0])
plt.plot(points, probs)
plt.grid()
plt.ylim([0, None])
plt.tight_layout()
plt.show()

# evaluate and plot the 2D marginal for the first and second dimensions
points, probs = grid.get_marginal([0, 1])
plot_marginal_2d(points=points, probabilities=probs, labels=["x", "y"])

# evaluate and plot the 2D marginal for the first and third dimensions
points, probs = grid.get_marginal([0, 2])
plot_marginal_2d(points=points, probabilities=probs, labels=["x", "z"])

# plot the convergence information
grid.plot_convergence()
