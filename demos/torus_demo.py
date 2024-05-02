from numpy import sqrt, array, exp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ridge import Ridge


class ToroidalGaussian:
    def __init__(self):
        self.R0 = 1.0  # major radius of the torus
        self.eps = 0.10  # aspect ratio of the torus
        self.coeff = -0.5 / (self.R0 * self.eps) ** 2

    def __call__(self, theta):
        x, y, z = theta
        r_sqr = z**2 + (sqrt(x**2 + y**2) - self.R0) ** 2
        return self.coeff * r_sqr


# set up the test-case posterior
posterior = ToroidalGaussian()

# specify settings for the grid
grid_spacing = array([0.04, 0.04, 0.02])
grid_centre = array([0.0, 0.0, 0.0])
grid_bounds = array([[-1.5, -1.5, -0.5], [1.5, 1.5, 0.5]]).T

# create a PdfGrid instance
grid = Ridge(
    spacing=grid_spacing,
    offset=grid_centre,
    bounds=grid_bounds
)

# evaluate the posterior
grid.evaluate_posterior(posterior=posterior)

# evaluate and plot the marginal for the first dimension
points, probs = grid.get_marginal([0])
plt.plot(points, probs)
plt.fill_between(points, 0.0, probs, color="C0", alpha=0.1)
plt.xlabel("x")
plt.ylabel("probability density")
plt.title("Marginal distribution for 'x' variable")
plt.grid()
plt.ylim([0, None])
plt.tight_layout()
plt.show()

# We can also use the matrix_plot method to plot all 1D and 2D marginal distributions
grid.matrix_plot(labels=["x", "y", "z"], colormap="viridis")

# plot the convergence information
grid.plot_convergence()

# we can also generate a sample
sample = grid.generate_sample(5000)

# plot the samples, coloring each point based on its probability
probs = array([posterior(s) for s in sample])
pnt_colors = exp(probs - probs.max())

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)
ax.set_zlabel("z", fontsize=15)
ax.scatter(*sample.T, c=pnt_colors)
plt.tight_layout()
plt.show()
