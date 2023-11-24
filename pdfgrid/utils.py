from numpy import append, delete, floor, ndarray, zeros


def neighbour_vectors(n: int, dtype, cutoff=2, include_center=False) -> ndarray:
    """
    Generates nearest neighbour list offsets from center cell
    """
    NN = zeros([(3 ** n), n], dtype = dtype)

    for k in range(n):
        L = 3 ** k
        NN[:L, k] = -1
        NN[L: 2 * L, k] = 0
        NN[2 * L: 3 * L, k] = 1

        if k != n - 1:  # we replace the first instance of the pattern with itself
            for j in range(3 ** (n - 1 - k)):  # less efficient but keeps it simple
                NN[0 + j * (3 * L): (j + 1) * (3 * L), k] = NN[0: 3 * L, k]

    m = int(floor(((3 ** n) - 1.0) / 2.0))
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
        zeroarray = zeros((1, n), dtype = dtype)
        NN = append(NN, zeroarray, axis = 0)

    return NN