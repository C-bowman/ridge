from numpy import linspace, exp, pi
from numpy import sqrt, array, eye, ndarray
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt

from ridge import Ridge


"""
Test case using a 4D correlated multivariate normal distribution
defined using a Gaussian-process.
"""


class GaussianProcessPosterior:
    def __init__(self, dimensions: int, scale: float = 1.0):
        self.n = dimensions
        self.x = linspace(1, self.n, self.n)
        self.L = scale

        dx = self.x[:, None] - self.x[None, :]
        self.K = exp(-0.5 * dx ** 2 / self.L ** 2)

        L = cholesky(self.K)
        self.iK = solve_triangular(L, eye(self.n), lower=True)
        self.iK = self.iK.T @ self.iK

    def __call__(self, theta: ndarray) -> float:
        return -0.5 * theta.T @ self.iK @ theta


# set up the test-case posterior
dims = 4
posterior = GaussianProcessPosterior(dimensions=dims, scale=1.5)

# specify settings for the grid
grid_spacing = array([0.2] * dims)
grid_centre = array([0.] * dims)
grid_bounds = array([[-8.0] * dims, [8.0] * dims]).T

# create a PdfGrid instance
grid = Ridge(
    spacing=grid_spacing,
    offset=grid_centre,
    bounds=grid_bounds,
    convergence=0.01
)

# evaluate the posterior
grid.evaluate_posterior(posterior=posterior)

# evaluate the marginal for the first dimension
points, probs = grid.get_marginal([0])

# evaluate the marginal analytically for comparison
axis = linspace(-20, 20, 41) * 0.2
exact_marginal = exp(-0.5 * axis**2) / sqrt(2 * pi)

# plot the marginal to verify the result
plt.plot(points, probs, label="RIDGE marginal")
plt.plot(axis, exact_marginal, marker="o", ls="none", markerfacecolor="none", label="exact marginal")
plt.grid()
plt.legend()
plt.ylim([0, None])
plt.xlabel("x0")
plt.ylabel("probability density")
plt.tight_layout()
plt.show()

# We can also use the matrix_plot method to plot all 1D and 2D marginal distributions
grid.matrix_plot(labels=["x0", "x1", "x2", "x3"])

# plot the convergence information
grid.plot_convergence()
