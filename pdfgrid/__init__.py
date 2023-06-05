from numpy import sqrt, log, exp, round, abs, floor
from numpy import array, zeros, arange, delete, append, linspace, frombuffer, stack
from numpy import argmax, where
from numpy import int16, ndarray
from numpy.random import normal
from copy import copy
from itertools import product
import sys
from pdfgrid.plotting import plot_convergence


class PdfGrid:
    """
    Adaptive grid evaluation for PDFs

    :param spacing: \
        A numpy ``ndarray`` specifying the grid spacing in each dimension.

    :param offset: \
        A numpy ``ndarray`` specifying the parameter values at the grid origin.

    :param search_scale: \
        The standard deviation (in grid cells) of the normal distribution used to
        randomly select cells for evaluation when searching for maxima.

    :param convergence: \
        The threshold for the fractional change in total probability which is used
        to determine when the algorithm has converged.
    """
    def __init__(self, spacing: ndarray, offset: ndarray, search_scale=5.0, convergence=1e-3):

        self.spacing = spacing if isinstance(spacing, ndarray) else array(spacing)
        self.offset = offset if isinstance(offset, ndarray) else array(offset)

        if self.spacing.ndim != 1 or self.offset.ndim != 1:
            raise ValueError(
                f"[ PdfGrid error ] \n \
                >> 'spacing' and 'offset' must be 1D numpy arrays, but have \
                >> dimensions {self.spacing.ndim} and {self.offset.ndim} respectively.\
                "
            )

        if self.spacing.size != self.offset.size:
            raise ValueError(
                f"[ PdfGrid error ] \n \
                >> 'spacing' and 'offset' must be 1D numpy arrays of equal size, but \
                >> have sizes {self.spacing.size} and {self.offset.size} respectively.\
                "
            )

        # CONSTANTS
        self.n_dims = self.spacing.size  # number of parameters / dimensions
        self.type = int16
        self.CC = zeros(self.n_dims, dtype=self.type)  # Set current vector as [0,0,0,...]
        self.NN = self.nn_vectors(self.n_dims)  # list of nearest-neighbour vectors
        self.n_neighbours = self.NN.shape[0]  # number of nearest-neighbours

        # SETTINGS
        self.threshold = 1
        self.threshold_adjust_factor = sqrt(0.5) ** self.n_dims
        self.climb_threshold = 5
        self.search_max = int(50 * self.n_dims)
        self.search_scale = search_scale
        self.convergence = convergence

        # DATA STORAGE
        self.coordinates = list()
        self.probability = list()

        # DECISION MAKING
        self.evaluated = set()
        self.exterior = list()
        self.to_evaluate = list()
        self.edge_push = list()
        self.total_prob = [0]
        self.state = "climb"
        self.globalmax = -1e10
        self.search_count = 0
        self.CC_index = 0
        self.fill_setup = True  # a flag for setup code which only must be run once

        # map to functions for proposing new evaluations in various states
        self.proposal_actions = {
            "climb": self.climb_proposal,
            "find": self.find_proposal,
            "fill": self.fill_proposal,
            "end": self.end,
        }

        # map to functions for updating cell information in various states
        self.update_actions = {
            "climb": self.climb_update,
            "find": self.find_update,
            "fill": self.fill_update,
            "end": self.end,
        }

        self.threshold_evals = [0]
        self.threshold_probs = [0]
        self.threshold_levels = [0]

        # DIAGNOSTICS
        self.verbose = True
        self.cell_batches = list()

        # Append the entire NN list to make the first evaluation list
        self.to_evaluate.append(self.CC)
        for i in range(self.n_neighbours):
            self.to_evaluate.append(self.CC + self.NN[i, :])

    def get_parameters(self) -> ndarray:
        """
        Get the parameter vectors for which the posterior log-probability needs to be
        calculated and passed to the ``give_probabilities`` method.

        :return: \
            A 2D numpy ``ndarray`` of parameter vectors with shape (n_vectors, n_dimensions).
        """
        return stack(self.to_evaluate) * self.spacing[None, :] + self.offset[None, :]

    def give_probabilities(self, log_probabilities: ndarray):
        """
        Accepts the newly-evaluated log-probabilities values corresponding to the
        parameter vectors given by the ``get_parameters`` method.
        """

        # Sum the incoming probabilities, add to running integral and append to integral array
        pmax = log_probabilities.max()
        self.total_prob.append(
            self.total_prob[-1] + exp(pmax + log(exp(log_probabilities - pmax).sum()))
        )

        # Here we convert the self.to_evaluate values to strings such
        # that they are hashable and can be added to the self.evaluated set.
        self.evaluated |= {v.tobytes() for v in self.to_evaluate}
        # now update the lists which store cell information
        self.probability.extend(log_probabilities)
        self.coordinates.extend(self.to_evaluate)
        self.exterior.extend([True] * log_probabilities.size)

        # run the state-specific update code
        self.update_actions[self.state](log_probabilities)
        # For diagnostic purposes, we save here the latest number of evals
        self.cell_batches.append(len(log_probabilities))
        # clean out the to_evaluate list here
        self.to_evaluate.clear()
        # generate a new set of evaluations
        if self.verbose:
            self.print_status()
        self.proposal_actions[self.state]()

    def fill_update(self, log_probabilities: ndarray):
        # add cells that are higher than threshold to edge_push
        prob_cutoff = self.globalmax - self.threshold
        self.edge_push = [
            v for v, p in zip(self.to_evaluate, log_probabilities) if p > prob_cutoff
        ]

        # if there are no cells above threshold, so lower it (or terminate)
        if len(self.edge_push) == 0:
            self.adjust_threshold()

            if self.threshold_probs[-2] == 0.0:
                delta_ptot = 1.0
            else:
                delta_ptot = (
                    self.threshold_probs[-1] - self.threshold_probs[-2]
                ) / self.threshold_probs[-2]

            if delta_ptot < self.convergence:
                self.state = "end"
                offset = self.threshold_evals[0]
                self.threshold_evals = [i - offset for i in self.threshold_evals]

    def climb_update(self, log_probabilities: ndarray):
        curr_prob = self.probability[self.CC_index]

        self.exterior[self.CC_index] = False
        # if current probability is greater than all nearest neighbours, it is local maximum
        if curr_prob > log_probabilities.max():
            # update global maximum probability if needed
            if curr_prob > self.globalmax:
                self.globalmax = curr_prob

            self.state = "find"  # moves to find state
        # otherwise, choose biggest neighbour as next current cell
        else:
            loc = argmax(log_probabilities)
            self.CC = self.to_evaluate[loc]
            # need to double check this
            self.CC_index = len(self.probability) - len(log_probabilities) + loc

    def find_update(self, *args):
        # if the last search evaluation has high enough probability, switch to the
        # 'climb' state and update the current cell info to tbe the last evaluation.
        if self.probability[-1] > (self.globalmax - self.climb_threshold):
            self.state = "climb"
            self.CC_index = len(self.probability) - 1
            # TODO - should self.CC also be set here?

    def check_neighbours(self):
        # find which neighbours of the current cell are unevaluated
        byte_strings = [v.tobytes() for v in self.CC + self.NN]
        return [i for i, s in enumerate(byte_strings) if s not in self.evaluated]

    def list_empty_neighbours(self, empty_NN):
        # Use the list generated by self.check_neighbours to list empty neighbours for evaluation
        self.to_evaluate.extend([self.CC + self.NN[i, :] for i in empty_NN])

    def climb_proposal(self):
        empty_NN = self.check_neighbours()
        if len(empty_NN) == 0:
            self.state = "find"
            self.find_proposal()
        else:
            self.list_empty_neighbours(empty_NN)

    def find_proposal(self):
        if self.search_count <= self.search_max:
            self.random_coordinate()
        else:
            prob_cutoff = self.globalmax - self.threshold

            for i in arange(len(self.probability))[::-1]:
                if self.probability[i] < prob_cutoff:
                    temp = self.coordinates[i].tobytes()
                    self.evaluated.remove(temp)
                    del self.probability[i]
                    del self.coordinates[i]
                    del self.exterior[i]

            self.state = "fill"
            self.fill_proposal()

    def fill_proposal(self):
        if self.fill_setup:
            # The very first time we get to fill, we need to locate all
            # relevant edge cells, i.e. those which have unevaluated neighbours
            # and are above the threshold.
            # TODO - the probability check may be unnecessary as all cells below threshold are removed earlier
            prob_cutoff = self.globalmax - self.threshold
            iterator = zip(self.coordinates, self.exterior, self.probability)
            edge_vecs = array(
                [v for v, ext, p in iterator if ext and p > prob_cutoff],
                dtype=self.type,
            )
            self.fill_setup = False
        else:
            edge_vecs = array(self.edge_push, dtype=self.type)

        # generate an array of all neighbours of all edge positions using outer addition via broadcasting
        r = (edge_vecs[None, :, :] + self.NN[:, None, :]).reshape(
            edge_vecs.shape[0] * self.NN.shape[0], self.n_dims
        )
        # treating the 2D array of vectors as an iterable returns
        # each column vector in turn.
        fill_set = {v.tobytes() for v in r}
        # now we have the set, we can use difference update to
        # remove all the index vectors which are already evaluated
        fill_set.difference_update(self.evaluated)

        # provision for all outer cells having been evaluated, so no
        # viable nearest neighbours - use full probability distribution
        # to find all edge cells (ie. lower than threshold)
        if len(fill_set) == 0:
            self.adjust_threshold()
            self.take_step()
        else:
            # here the set of fill vectors is converted back to an array
            self.to_evaluate = [frombuffer(s, dtype=self.type) for s in fill_set]

    def end(self):
        pass

    def adjust_threshold(self):
        """
        Adjust the threshold to a new value that is threshold + threshold_adjust_factor
        """
        # first collect stats
        self.threshold_levels.append(copy(self.threshold))
        self.threshold_probs.append(self.total_prob[-1])
        self.threshold_evals.append(len(self.probability))

        prob_cutoff = self.globalmax - self.threshold
        self.edge_push = [
            v for v, p in zip(self.coordinates, self.probability) if p < prob_cutoff
        ]
        self.threshold += self.threshold_adjust_factor

    def random_coordinate(self):
        """
        <purpose>
            Uses a Monte-Carlo approach to search for unevaluated
            grid cells.
            Once an empty cell is located, it is added to the
            to_evaluate list.
        """
        while True:
            # pick set of normally distributed random indices
            self.CC = (round(normal(scale=self.search_scale, size=self.n_dims))).astype(
                self.type
            )

            # check if the new cell has already been evaluated
            byte_str = self.CC.tobytes()
            if byte_str not in self.evaluated:
                self.to_evaluate.append(self.CC)
                self.search_count += 1
                break

    def print_status(self):
        msg = f"\r [ {len(self.probability)} total evaluations, state is {self.state} ]"
        sys.stdout.write(msg)
        sys.stdout.flush()

    def nn_vectors(self, n, cutoff=1, include_center=False):
        """
        Generates nearest neighbour list offsets from center cell
        """
        NN = zeros([(3**n), n], dtype=self.type)

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
            zeroarray = zeros((1, self.n_dims), dtype=self.type)
            NN = append(NN, zeroarray, axis=0)

        return NN

    def plot_convergence(self):
        plot_convergence(self.threshold_evals, self.threshold_probs)

    def get_marginal(self, z):
        """
        Integrate over the dimensions of the parameter space that are not
         requested in z

        Input - z - a list of one or more integers indicating the dimension(s)
                    over which NOT to integrate.  Integrate the rest.
        """
        I = array(self.coordinates)
        probs = array(self.probability)
        probs -= probs.max()

        # 1D code
        if type(z) is int:
            # find range of the 1D grid
            lwr = I[:, z].min()
            upr = I[:, z].max()

            # make the grid
            L = upr - lwr + 1
            i_grid = linspace(lwr, upr, L, dtype=self.type)
            P_grid = zeros(L)

            for i, j in enumerate(i_grid):
                bools = I[:, z] == j
                if bools.any():
                    P_grid[i] = exp(probs[bools]).sum()

        # TODO - should check whether the n-dimensional code can be overhauled
        else:  # multi-dimensional code
            m = len(z)
            lwr = zeros(m)  # list of starting indices
            upr = zeros(m)  # list of ending indices
            L = list()
            R = list()
            for i in range(m):
                lwr[i] = int(I[:, z[i]].min())
                upr[i] = int(I[:, z[i]].max())
                L.append(int(upr[i] - lwr[i]))  # list of lengths
                R.append(range(int(upr[i] - lwr[i])))
                # NOTE there should be a +1 above, but something
                # else is the wrong length somewhere... needs checking

            # i_grid is a list of indices axes for each parameter
            i_grid = list()
            for i in range(m):
                i_grid.append(arange(lwr[i], upr[i]))

            # now make the grid to hold the marginal distribution
            P_grid = zeros(L)
            # now use cartesian product to create list of all indices
            ind_list = product(*R)

            # because only one coordinate is changed at a time, we could
            # make this more efficient by storing the different boole
            # vectors, but not needed unless this becomes a computational
            # bottleneck.
            for k in ind_list:
                booles = I[:, z[0]] == (i_grid[0])[k[0]]
                for j in range(1, m):
                    booles = booles & (I[:, z[j]] == (i_grid[j])[k[j]])

                if any(booles):
                    inds = where(booles)
                    P_grid[k] = exp(probs[inds]).sum()
                else:
                    P_grid[k] = 0

        return i_grid, P_grid
