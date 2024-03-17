from numpy import append, delete, floor, ndarray, zeros, unique, int16
from numpy.random import default_rng


def neighbour_vectors(n: int, dtype, cutoff=1, include_center=False) -> ndarray:
    """
    Generates nearest neighbour list offsets from center cell
    """
    NN = zeros([(3**n), n], dtype=dtype)

    for k in range(n):
        L = 3**k
        NN[:L, k] = -1
        NN[L : 2 * L, k] = 0
        NN[2 * L : 3 * L, k] = 1

        if k != n - 1:  # we replace the first instance of the pattern with itself
            for j in range(3 ** (n - 1 - k)):  # less efficient but keeps it simple
                NN[0 + j * (3 * L) : (j + 1) * (3 * L), k] = NN[0 : 3 * L, k]

    m = int(floor(((3**n) - 1.0) / 2.0))
    NN = delete(NN, m, 0)

    # Euclidian distance neighbour trimming
    if cutoff:
        cut_list = list()
        for i in range(len(NN[:, 0])):
            temp = abs(NN[i, :]).sum()
            if temp > cutoff:
                cut_list.append(i)

        for i in cut_list[::-1]:
            NN = delete(NN, i, 0)

    if include_center:
        zeroarray = zeros((1, n), dtype=dtype)
        NN = append(NN, zeroarray, axis=0)

    return NN


def uniform_grid_sample(
    lower_bounds: ndarray, upper_bounds: ndarray, n_samples: int, n_dims: int, seed=None
) -> ndarray:
    # find total number of cells within the bounds
    total_cells = (upper_bounds - lower_bounds + 1).prod()
    # ensure number of samples doesn't exceed available cells
    n_samples = min(total_cells, n_samples)
    # calculate the expected number of duplicates
    expected_collisions = n_samples * (
        1 - ((total_cells - 1) / total_cells) ** (n_samples - 1)
    )
    # sample the cell coordinates
    extra_samples = round(expected_collisions + 1) * 3
    rng = default_rng() if seed is None else default_rng(seed)
    samples = rng.integers(
        low=lower_bounds,
        high=upper_bounds,
        endpoint=True,
        size=[n_samples + extra_samples, n_dims],
        dtype=int16,
    )
    # remove any duplicates
    return unique(samples, axis=0)[:n_samples, :]


def compute_marginal(
    coords: ndarray, probs: ndarray, spacing: ndarray, offset: ndarray, z: list[int]
) -> tuple[ndarray, ndarray]:
    """
    Calculate the marginal distribution for given variables.

    :return points, probabilities: \
        The points at which the marginal distribution is evaluated, and the
        associated marginal probability density.
    """
    # find all unique sub-vectors for the marginalisation dimensions and their indices
    uniques, inverse, counts = unique(
        coords[:, z], return_inverse=True, return_counts=True, axis=0
    )
    # use the indices and the counts to calculate the CDF then convert to the PDF
    marginal_pdf = probs[inverse.argsort()].cumsum()[counts.cumsum() - 1]
    marginal_pdf[1:] -= marginal_pdf[:-1]
    # use the spacing to properly normalise the PDF
    marginal_pdf /= spacing[z].prod() * marginal_pdf.sum()
    # convert the coordinate vectors to parameter values
    uniques = uniques * spacing[None, z] + offset[None, z]
    return uniques.squeeze(), marginal_pdf
