from numpy import sqrt, array
import matplotlib.pyplot as plt
from pdfgrid import PdfGrid


class ToroidalGaussian(object):
    def __init__(self):
        self.R0 = 1.
        self.eps = 0.05
        self.w2 = (self.R0*self.eps)**2

    def __call__(self, theta):
        x, y, z = theta
        r = sqrt(z**2 + (sqrt(x**2 + y**2) - self.R0)**2)
        return -0.5*r**2 / self.w2


posterior = ToroidalGaussian()
grid_spacing = array([0.05, 0.05, 0.02])
SPG = PdfGrid(3)

# Main GridFill loop
from time import time
t1 = time()
while SPG.state != "end":
    P = [posterior(theta*grid_spacing) for theta in SPG.to_evaluate]
    SPG.update_cells(P)
    SPG.take_step()
t2 = time()

print(f"\n # RUNTIME: {(t2-t1)*1000:.1f} ms")

indices, probs = SPG.get_marginal(0)
plt.plot(indices*grid_spacing[0], probs)
plt.grid()
plt.tight_layout()
plt.show()

params = [0,1]
ind_axes, probs = SPG.get_marginal(params)

ax1, ax2 = [ax*grid_spacing[p] for ax, p in zip(ind_axes, params)]
plt.contourf(ax1, ax2, probs.T)
plt.show()

SPG.plot_convergence()