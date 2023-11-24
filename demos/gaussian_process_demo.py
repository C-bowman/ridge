from numpy import linspace, exp, pi
from numpy import sqrt, array, eye
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular
from scipy.integrate import trapz
import matplotlib.pyplot as plt

from pdfgrid import PdfGrid
from pdfgrid.plotting import plot_marginal_2d


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

    def __call__(self, theta):
        return -0.5 * theta.T @ self.iK @ theta


dims = 4
posterior = GaussianProcessPosterior(dimensions=dims, scale=1.5)
grid_spacing = array([0.2] * dims)
grid_centre = array([0.] * dims)
grid_bounds = array([[-8.0] * dims, [8.0] * dims]).T


grid = PdfGrid(
    spacing=grid_spacing,
    offset=grid_centre,
    bounds=grid_bounds,
    convergence=0.02
)

while grid.state != "end":
    P = array([posterior(theta) for theta in grid.get_parameters()])
    grid.give_probabilities(P)

points, probs = grid.get_marginal([0])

plt.plot(points, probs, label="pdf-grid marginal")
axis = linspace(-20, 20, 41) * 0.2
exact_marginal = exp(-0.5 * axis**2) / sqrt(2 * pi)
plt.plot(axis, exact_marginal, marker="o", ls="none", markerfacecolor="none", label="exact marginal")

plt.grid()
plt.legend()
plt.ylim([0, None])
plt.tight_layout()
plt.show()

points, probs = grid.get_marginal([0, 1])
plot_marginal_2d(points, probs)

grid.plot_convergence()