"""
QOPeriod with GCD extraction
"""

import numpy as np
from functools import reduce
from itertools import combinations
from pyPeriod import QOPeriods
from .utils import phi, get_factors


class QOPeriodsWithGCDsExtracted(QOPeriods):
    """
    A subclass of QOPeriods that modifies the method for generating subspaces to account for the greatest common divisor (GCD) of pairs of periods.

    This class overrides the `get_subspaces` method of the parent class `QOPeriods` to generate subspaces in a way that takes into account the GCD of pairs of periods. This can be useful in signal processing tasks where the periods of the signal have a common factor.

    Parameters
    ----------
    basis_type : str, optional
        The type of basis to use for the subspaces. Can be "natural" or "ramanujan". The default is "natural".

    trunc_to_integer_multiple : bool, optional
        If True, the signal will be truncated to an integer multiple of the period. The default is False.

    Methods
    -------
    get_subspaces(Q, N):
        Generate the subspaces for all periods in Q, taking into account the GCD of pairs of periods.
    """

    def __init__(self, basis_type="natural", trunc_to_integer_multiple=False):
        super(QOPeriodsWithGCDsExtracted, self).__init__(
            basis_type, trunc_to_integer_multiple
        )  # no need to orthogonalize since we extract the GCDs (i.e. everything is guarenteed to be orthogonal in get_subspaces)

    def get_subspaces(self, Q: set, N: int) -> tuple:
        """
        Generate the subspaces for all periods in Q, taking into account the GCD of pairs of periods.

        This method overrides the parent class method to generate subspaces in a way that takes into account the GCD of pairs of periods. This involves generating a set of periods that includes the GCDs of all pairs of periods in Q, and then generating the subspaces for these periods. Example: Q = {8, 12, 15}. The GCDs of all pairs of periods in Q are {1, 2, 3, 4, 6}. Therefore, the set of periods for which to generate subspaces is then {1, 2, 3, 4, 6, 8, 12, 15}. The subspaces for these periods are then generated and the number of rows retained for each period is equal to the Euler's toitent of the period. The number of rows retained for each period is then stored in a dictionary.

        Parameters
        ----------
        Q : set
            A set of periods for which to generate the subspaces.

        N : int
            The length of the signal for which the subspaces are being generated.

        Returns
        -------
        A : ndarray
            The 'basis matrix' that is passed to solve_quadratic. This is a 2D array where each row corresponds to a subspace.

        d : dict
            A dictionary that holds information about A where the key-value pair is (q, rows) where q is the period and rows is the number of rows retained.
        """
        P = set()  # Initialize an empty set for periods
        pairs = list(
            combinations(Q, 2)
        )  # Generate all pairs of periods (Cartesian product)

        # For each pair of periods, extract the GCD and add it to the set of periods
        for pair in pairs:
            intersection = set.intersection(
                get_factors(pair[0]), get_factors(pair[1])
            )  # Compute the intersection of the factors of the two periods
            P = P.union(intersection)  # Add the intersection to the set of periods

        P = P.union(Q)  # Add the original periods to the set of periods
        P = set(sorted(P))  # Sort the set of periods
        A = np.array([]).reshape((0, N))  # Initialize an empty basis matrix
        d = (
            {}
        )  # Initialize an empty dictionary for storing information about the basis matrix

        # For each period in the set of periods
        for p in P:
            F = get_factors(
                p, remove_n=True
            )  # Get the factors of the period, removing the period itself
            F = F.intersection(
                P
            )  # Compute the intersection of the set of factors with the set of periods
            keep = int(
                p - np.sum(np.array([phi(f) for f in F]))
            )  # Compute the number of rows to keep for this period
            # ss = self.Pp(int(p), N, keep, self._basis_type) * np.sqrt(p) # Compute the subspace for this period
            ss = self.Pp(
                int(p), N, keep, self._basis_type
            )  # Compute the subspace for this period
            A = np.vstack((A, ss))  # Add the subspace to the basis matrix
            d[str(p)] = (
                keep  # Add the number of rows kept for this period to the dictionary
            )

        return (A, d)
