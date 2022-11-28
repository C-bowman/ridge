import matplotlib.pyplot as plt
from numpy import array


def plot_convergence(evaluations, probabilities):
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121)
    ax1.plot(evaluations, probabilities, ".-", lw=2)
    ax1.set_xlabel("total posterior evaluations")
    ax1.set_ylabel("total probability of evaluated cells")
    ax1.grid()

    ax2 = fig.add_subplot(122)
    p = array(probabilities[1:])
    frac_diff = p[1:] / p[:-1] - 1
    ax2.plot(evaluations[2:], frac_diff, alpha=0.5, lw=2, c="C0")
    ax2.plot(evaluations[2:], frac_diff, "D", c="C0")
    ax2.set_xlim([0.0, None])
    ax2.set_yscale("log")
    ax2.set_xlabel("total posterior evaluations")
    ax2.set_ylabel("fractional change in total probability")
    ax2.grid()

    plt.tight_layout()
    plt.show()
