from numpy import sqrt, array
import matplotlib.pyplot as plt
from ridge import Ridge


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
grid = Ridge(
    spacing=grid_spacing,
    offset=grid_centre,
    bounds=grid_bounds,
    convergence=0.02
)

# evaluate the posterior
grid.evaluate_posterior(posterior=posterior)

# evaluate and plot the marginal for the first dimension
points, probs = grid.get_marginal([0])
plt.plot(points, probs)
plt.grid()
plt.ylim([0, None])
plt.tight_layout()
plt.show()

# We can also use the matrix_plot method to plot all 1D and 2D marginal distributions
grid.matrix_plot(labels=["x", "y", "z"], colormap="viridis")

# plot the convergence information
grid.plot_convergence()
