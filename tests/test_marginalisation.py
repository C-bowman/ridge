from numpy import linspace, exp, pi, cov
from numpy import sqrt, array, eye, ndarray
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular
from ridge import Ridge


class GaussianProcessPosterior:
    def __init__(self, dimensions: int, scale: float = 1.0):
        self.n = dimensions
        self.x = linspace(1, self.n, self.n)
        self.L = scale

        dx = self.x[:, None] - self.x[None, :]
        self.K = exp(-0.5 * dx**2 / self.L**2)

        L = cholesky(self.K)
        self.iK = solve_triangular(L, eye(self.n), lower=True)
        self.iK = self.iK.T @ self.iK

    def __call__(self, theta: ndarray) -> float:
        return -0.5 * theta.T @ self.iK @ theta


def test_marginalisation():
    # set up the test-case posterior
    dims = 4
    posterior = GaussianProcessPosterior(dimensions=dims, scale=1.5)

    # specify settings for the grid
    grid_spacing = array([0.2] * dims)
    grid_centre = array([0.0] * dims)
    grid_bounds = array([[-8.0] * dims, [8.0] * dims]).T

    # create a Ridge instance
    grid = Ridge(
        spacing=grid_spacing,
        offset=grid_centre,
        bounds=grid_bounds,
        convergence_threshold=0.01
    )

    # evaluate the posterior
    grid.evaluate_posterior(posterior=posterior)

    # evaluate the marginal for the first dimension
    points, probs = grid.get_marginal([0])

    # evaluate the marginal analytically for comparison
    exact_marginal = exp(-0.5 * points**2) / sqrt(2 * pi)

    # verify the computed marginal agrees with the analytic result
    error = (probs - exact_marginal) / exact_marginal.max()
    assert abs(error).max() < 0.01


def test_generate_samples():
    # set up the test-case posterior
    dims = 2
    posterior = GaussianProcessPosterior(dimensions=dims, scale=1.5)

    # specify settings for the grid
    grid_spacing = array([0.05] * dims)
    grid_centre = array([0.0] * dims)
    grid_bounds = array([[-8.0] * dims, [8.0] * dims]).T

    # create a Ridge instance
    grid = Ridge(
        spacing=grid_spacing,
        offset=grid_centre,
        bounds=grid_bounds,
        convergence_threshold=0.01
    )

    # evaluate the posterior
    grid.evaluate_posterior(posterior=posterior)
    # generate the samples
    samples = grid.generate_samples(n_samples=50000)
    # test the mean and covariance of the sample against the true values
    covariance_error = cov(samples.T) - posterior.K
    assert (abs(covariance_error) < 0.03).all()
    assert (abs(samples.mean(axis=0)) < 0.03).all()
