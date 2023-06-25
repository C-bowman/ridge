from numpy import sqrt, log, exp, floor, ceil
from numpy import array, zeros, frombuffer, stack
from numpy import unique
from numpy import int16, ndarray
from numpy.random import default_rng
from copy import copy
import sys
from pdfgrid.plotting import plot_convergence
from pdfgrid.utils import neighbour_vectors
rng = default_rng()


class PdfGrid:
    """
    Adaptive grid evaluation for PDFs

    :param spacing: \
        A numpy ``ndarray`` specifying the grid spacing in each dimension.

    :param offset: \
        A numpy ``ndarray`` specifying the parameter values at the grid origin.

    :param convergence: \
        The threshold for the fractional change in total probability which is used
        to determine when the algorithm has converged.
    """
    def __init__(self,
        spacing: ndarray,
        offset: ndarray,
        bounds=None,
        convergence=1e-3
    ):

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

        if bounds is not None:
            assert bounds.ndim == 2
            assert bounds.shape == (self.spacing.size, 2)
            assert (bounds[:, 0] < bounds[:, 1]).all()
            self.lower_bounds = ceil((bounds[:, 0] - self.offset) / self.spacing).astype(int16)
            self.upper_bounds = floor((bounds[:, 1] - self.offset) / self.spacing).astype(int16)
            assert (self.lower_bounds < self.upper_bounds).all()

        # CONSTANTS
        self.n_dims = self.spacing.size  # number of parameters / dimensions
        self.current_cell = zeros(self.n_dims, dtype=int16)  # Set current vector as [0,0,0,...]
        self.neighbours = neighbour_vectors(self.n_dims, int16)  # nearest-neighbour vectors
        self.n_neighbours = self.neighbours.shape[0]  # number of nearest-neighbours

        # SETTINGS
        self.threshold = 1
        self.threshold_adjust_factor = sqrt(0.5) ** self.n_dims
        self.n_samples = 50 * 2 ** (self.n_dims - 1)
        self.n_climbs = self.n_samples // 10
        self.convergence = convergence

        # DATA STORAGE
        self.coordinates = []
        self.probability = []

        # DECISION MAKING
        self.evaluated = set()
        self.exterior = []
        self.edge_push = []
        self.total_prob = [0]
        self.state = "sampling"
        self.max_prob = -1e100
        self.current_index = 0
        self.fill_setup = True  # a flag for setup code which only must be run once

        # map to functions for updating cell information in various states
        self.update_actions = {
            "sampling": self.sampling_update,
            "climb": self.climb_update,
            "fill": self.fill_update,
        }

        self.threshold_evals = [0]
        self.threshold_probs = [0]
        self.threshold_levels = [0]

        # DIAGNOSTICS
        self.report_progress = True
        self.cell_batches = list()

        # find total number of cells within the bounds
        total_cells = (self.upper_bounds - self.lower_bounds - 1).prod()
        # ensure number of samples doesn't exceed available cells
        self.n_samples = min(total_cells, self.n_samples)
        # calculate the expected number of duplicates
        expected_collisions = self.n_samples * (1 - ((total_cells - 1) / total_cells) ** (self.n_samples - 1))
        # sample the cell coordinates
        extra_samples = round(expected_collisions + 1) * 3
        samples = rng.integers(
            low=self.lower_bounds,
            high=self.upper_bounds,
            endpoint=True,
            size=[self.n_samples + extra_samples, self.n_dims],
            dtype=int16
        )
        # remove any duplicates and create the evaluation coordinates
        self.to_evaluate = unique(samples, axis=0)[:self.n_samples, :]

    def get_parameters(self) -> ndarray:
        """
        Get the parameter vectors for which the posterior log-probability needs to be
        calculated and passed to the ``give_probabilities`` method.

        :return: \
            A 2D numpy ``ndarray`` of parameter vectors with shape (n_vectors, n_dimensions).
        """
        return self.to_evaluate * self.spacing[None, :] + self.offset[None, :]

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

        if pmax > self.max_prob:
            self.max_prob = pmax

        # Here we convert the self.to_evaluate values to strings such
        # that they are hashable and can be added to the self.evaluated set.
        self.evaluated |= {v.tobytes() for v in self.to_evaluate}
        # now update the lists which store cell information
        self.probability.extend(log_probabilities)
        self.coordinates.extend(self.to_evaluate)
        self.exterior.extend([True] * log_probabilities.size)
        # For diagnostic purposes, we save here the latest number of evals
        self.cell_batches.append(len(log_probabilities))

        # run the state-specific update code
        self.update_actions[self.state](log_probabilities)

        if self.report_progress:
            self.print_status()

    def fill_update(self, log_probabilities: ndarray):
        # add cells that are higher than threshold to edge_push
        prob_cutoff = self.max_prob - self.threshold

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
                self.ending_cleanup()
                return

        self.fill_proposal()

    def climb_update(self, log_probabilities: ndarray):
        curr_prob = self.probability[self.current_index]
        self.exterior[self.current_index] = False

        # if a neighbour has larger probability, move the current cell there
        if curr_prob < log_probabilities.max():
            loc = log_probabilities.argmax()
            self.current_cell = self.to_evaluate[loc, :]
            self.current_index = len(self.probability) - len(log_probabilities) + loc
            assert (self.coordinates[self.current_index] == self.current_cell).all()

        # if the current cell is a local maximum, keep it, and it will
        # be switched for the next climbing start in the proposal:
        self.climb_proposal()

    def sampling_update(self, log_probabilities: ndarray):
        # create list of samples ordered so we can .pop() to get the highest prob
        n_climbs = min(self.n_climbs, log_probabilities.size - 1)
        inds = log_probabilities.argsort()[-n_climbs:]
        self.climb_starts = [(i, self.to_evaluate[i, :]) for i in inds]
        self.current_index, self.current_cell = self.climb_starts.pop()
        assert self.probability[self.current_index] == log_probabilities.max()

        # transition to climbing
        self.state = "climb"
        self.climb_proposal()

    def climb_proposal(self):
        while len(self.climb_starts) > 0:
            neighbour_set = {v.tobytes() for v in self.current_cell[None, :] + self.neighbours}
            neighbour_set -= self.evaluated

            if len(neighbour_set) == 0:
                self.current_index, self.current_cell = self.climb_starts.pop()
            else:
                self.to_evaluate = stack([frombuffer(s, dtype=int16) for s in neighbour_set])
                break
        else:
            self.state = "fill"
            self.fill_proposal()

    def fill_proposal(self):
        if self.fill_setup:
            # The very first time we get to fill, we need to locate all
            # relevant edge cells, i.e. those which have unevaluated neighbours
            # and are above the threshold.
            prob_cutoff = self.max_prob - self.threshold
            iterator = zip(self.coordinates, self.exterior, self.probability)
            edge_vecs = array(
                [v for v, ext, p in iterator if ext and p > prob_cutoff],
                dtype=int16,
            )
            self.fill_setup = False
        else:
            edge_vecs = array(self.edge_push, dtype=int16)

        # generate an array of all neighbours of all edge positions using outer addition via broadcasting
        r = (edge_vecs[None, :, :] + self.neighbours[:, None, :]).reshape(
            edge_vecs.shape[0] * self.neighbours.shape[0], self.n_dims
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
            raise ValueError("fill set empty")
        else:
            # here the set of fill vectors is converted back to an array
            self.to_evaluate = stack([frombuffer(s, dtype=int16) for s in fill_set])

    def adjust_threshold(self):
        """
        Adjust the threshold to a new value that is threshold + threshold_adjust_factor
        """
        # first collect stats
        self.threshold_levels.append(copy(self.threshold))
        self.threshold_probs.append(self.total_prob[-1])
        self.threshold_evals.append(len(self.probability))

        prob_cutoff = self.max_prob - self.threshold
        lower_lim = prob_cutoff - 2*self.threshold_adjust_factor
        self.edge_push = [
            v for v, p in zip(self.coordinates, self.probability) if lower_lim < p < prob_cutoff
        ]
        self.threshold += self.threshold_adjust_factor

    def ending_cleanup(self):
        inds = (array(self.probability) > (self.max_prob - self.threshold)).nonzero()[0]
        self.probability = [self.probability[i] for i in inds]
        self.coordinates = [self.coordinates[i] for i in inds]
        # clean up memory for decision-making data
        self.evaluated.clear()
        self.exterior.clear()
        self.edge_push.clear()
        self.to_evaluate = 0.

    def print_status(self):
        msg = f"\r [ {len(self.probability)} total evaluations, state is {self.state} ]"
        sys.stdout.write(msg)
        sys.stdout.flush()

    def plot_convergence(self):
        plot_convergence(self.threshold_evals, self.threshold_probs)

    def get_marginal(self, variables: list[int]) -> tuple[ndarray, ndarray]:
        """
        Calculate the marginal distribution for given variables.

        :param variables: \
            The indices of the variable(s) for which the marginal distribution is
            calculated, given as an integer or list of integers.

        :return points, probabilities: \
            The points at which the marginal distribution is evaluated, and the
            associated marginal probability density.
        """
        z = variables if isinstance(variables, list) else [variables]
        coords = stack(self.coordinates)
        probs = array(self.probability)
        probs = exp(probs - log(self.total_prob[-1]))
        # find all unique sub-vectors for the marginalisation dimensions and their indices
        uniques, inverse, counts = unique(coords[:, z], return_inverse=True, return_counts=True, axis=0)
        # use the indices and the counts to calculate the CDF then convert to the PDF
        marginal_pdf = probs[inverse.argsort()].cumsum()[counts.cumsum() - 1]
        marginal_pdf[1:] -= marginal_pdf[:-1]
        # use the spacing to properly normalise the PDF
        marginal_pdf /= self.spacing[z].prod()
        # convert the coordinate vectors to parameter values
        uniques = uniques * self.spacing[None, z] + self.offset[None, z]
        return uniques.squeeze(), marginal_pdf

    def generate_sample(self, n_samples: int) -> ndarray:
        """
        Generate samples by approximating the PDF using nearest-neighbour
        interpolation around the evaluated grid cells.

        :param n_samples: \
            Number of samples to generate.

        :return: \
            The samples as a 2D numpy ``ndarray`` with shape
            ``(n_samples, n_dimensions)``.
        """
        # normalise the probabilities
        p = array(self.probability)
        p = exp(p - p.max())
        p /= p.sum()
        # use the probabilities to weight samples of the grid cells
        indices = rng.choice(len(self.probability), size=n_samples, p=p)
        # gather the evaluated cell coordinates into a 2D numpy array
        params = stack(self.coordinates) * self.spacing[None, :] + self.offset[None, :]
        # Randomly pick points within the sampled cells
        sample = params[indices, :] + rng.uniform(
            low=-0.5*self.spacing,
            high=0.5*self.spacing,
            size=[n_samples, self.n_dims]
        )
        return sample
