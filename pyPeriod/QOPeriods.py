"""
    IMPORTANT!!
    You must use Python >= 3.6 for ordered dictionaries. Otherwise you are not
    guarenteed to get back the original periods from get_periods().
"""

import numpy as np
from .Periods import Periods
from functools import reduce
import itertools
import random
from scipy import linalg as spla
import warnings


def phi(n: int) -> int:
    """
    Compute Euler's totient function.

    Euler's totient function, denoted φ(n), counts the positive integers up to a given integer n that are relatively prime to n. In other words, it is the number of integers k in the range 1 ≤ k ≤ n for which the greatest common divisor gcd(n, k) is equal to 1.

    Parameters
    ----------
    n : int
        The input integer for which the totient function is to be calculated.

    Returns
    -------
    int
        The value of Euler's totient function for the input integer.

    Examples
    --------
    >>> phi(9)
    6
    >>> phi(10)
    4
    """
    amount = 0
    for k in range(1, n + 1):
        if np.gcd(n, k) == 1:
            amount += 1
    return amount


def get_factors(n: int, remove_1: bool = False, remove_n: bool = False) -> set:
    """
    Get all factors of a given number.

    Parameters
    ----------
    n : int
        The number to factor.
    remove_1 : bool, optional
        If True, remove 1 from the factors, by default False.
    remove_n : bool, optional
        If True, remove `n` from the factors, by default False.

    Returns
    -------
    set
        A set of all factors of the given number.
    """
    facs = set(
        reduce(
            list.__add__,
            ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
        )
    )
    # get rid of 1 and the number itself
    if remove_1:
        facs.remove(1)
    if remove_n and n != 1:
        facs.remove(n)
    return facs  # retuned as a set


def rms(x: list) -> float:
    return np.sqrt(np.sum(np.power(x, 2)) / len(x))


def flatten(t: list) -> list:
    return [item for sublist in t for item in sublist]


def reduce_rows(A: Any) -> Any:
    AA = A[0]
    rank = np.linalg.matrix_rank(AA)  # should be 1
    for row in A[1:]:
        aa = np.vstack((AA, row))
        if np.linalg.matrix_rank(aa) > int(rank):
            AA = aa
            rank = np.linalg.matrix_rank(aa)
    return AA


def get_primes(max: int = 1000000) -> list[int]:
    """
    Generate all prime numbers up to a given maximum. Resulting array does not include 1.

    Parameters
    ----------
    max : int, optional
        The maximum number up to which to generate primes, by default 1000000.

    Returns
    -------
    array_like
        An array of all prime numbers up to the given maximum.
    """
    primes = np.arange(3, max + 1, 2)
    isprime = np.ones((max - 1) // 2, dtype=bool)
    for factor in primes[: int(np.sqrt(max))]:
        if isprime[(factor - 2) // 2]:
            isprime[(factor * 3 - 2) // 2 :: factor] = 0
    return np.insert(primes[isprime], 0, 2)


def normalize(x: list, level: int = 1) -> np.ndarray[float]:
    """
    Normalize a list of numerical values. This function scales the input list so that its maximum absolute value is equal to the specified level.

    Parameters
    ----------
    x : list
        The list of numerical values to be normalized.

    level : int, optional
        The desired maximum absolute value for the normalized list. The default is 1.

    Returns
    -------
    list
        The normalized list of numerical values, scaled such that the maximum absolute value is equal to the specified level.

    Examples
    --------
    >>> normalize([1, 2, 3, 4, 5])
    [0.2, 0.4, 0.6, 0.8, 1.0]

    >>> normalize([-1, 0, 1], level=10)
    [-10, 0, 10]
    """
    m = np.max(np.abs(x))
    return (x / m) * level


class QOPeriods:
    PRIMES = set(
        get_primes(10000)
    )  # a class variable to hold a set of primes (does not include 1)

    def __init__(
        self, basis_type="natural", trunc_to_integer_multiple=False, orthogonalize=False
    ):
        """
        Initializes a Periods object.

        Parameters
        ----------
        basis_type : str, optional
            The type of basis to use. Defaults to "natural".
        trunc_to_integer_multiple : bool, optional
            Whether to truncate the basis functions to an integer multiple of their period. Defaults to False.
        orthogonalize : bool, optional
            Whether to orthogonalize the basis functions. Defaults to False.

        Attributes
        ----------
        _trunc_to_integer_multiple : bool
            Whether to truncate the basis functions to an integer multiple of their period.
        _output : None or ndarray
            Placeholder for storing the output of the function.
        _basis_type : str
            The type of basis being used.
        _verbose : bool
            Whether to display verbose output. Defaults to False.
        _k : int
            The number of basis functions to use.
        _orthogonalize : bool
            Whether to orthogonalize the basis functions.
        _window : bool
            Placeholder for whether to use a window function.
        _output_bases : None or ndarray
            Placeholder for storing the output bases.
        _container : list
            Placeholder for storing the container.

        """
        super(Periods, self).__init__(trunc_to_integer_multiple, orthogonalize)
        self._output = None
        self._basis_type = basis_type
        self._verbose = False
        self._k = 0
        self._window = False

        # for the test function
        self._output_bases = None
        self._container = []

    # @staticmethod
    # def project(
    #     data,
    #     p=2,
    #     trunc_to_integer_multiple=False,
    #     orthogonalize=False,
    #     return_single_period=False,
    # ):
    #     """
    #     Project a signal onto its periodic components.

    #     Parameters
    #     ----------
    #     data : ndarray
    #         Input signal to be projected
    #     p : int, optional
    #         The period of the signal, default is 2
    #     trunc_to_integer_multiple : bool, optional
    #         Flag to indicate if the signal is truncated to an integer multiple of the period, default is False
    #     orthogonalize : bool, optional
    #         Flag to indicate if the projection is orthogonalized, default is False
    #     return_single_period : bool, optional
    #         Flag to indicate if the single period projection is returned, default is False

    #     Returns
    #     -------
    #     projection : ndarray
    #         The projected signal

    #     Notes
    #     -----
    #     This function computes the periodic components of the input signal using the method presented in "Orthogonal,
    #     exactly periodic subspace decomposition" (D.D. Muresan, T.W. Parks), 2003. If the input is not truncated to an
    #     integer multiple of the period, a faster but equivalent method is used to compute the periodic components.
    #     If the orthogonalize flag is set to True, the projection is orthogonalized using the same method.
    #     """

    #     cp = data.copy()
    #     samples_short = int(
    #         np.ceil(len(cp) / p) * p - len(cp)
    #     )  # calc how many samples short for rectangle
    #     cp = np.pad(cp, (0, samples_short))  # pad it
    #     cp = cp.reshape(int(len(cp) / p), p)  # reshape it to a rectangle

    #     if trunc_to_integer_multiple:
    #         if samples_short == 0:
    #             single_period = np.mean(cp, 0)  # don't need to omit the last row
    #         else:
    #             single_period = np.mean(
    #                 cp[:-1], 0
    #             )  # just take the mean of the truncated version and output a single period
    #     else:
    #         ## this is equivalent to the method presented in the paper but significantly faster ##
    #         # do the mean manually. get the divisors from the last row since the last samples_short values will be one less than the others
    #         divs = np.zeros(cp.shape[1])
    #         for i in range(cp.shape[1]):
    #             if i < (cp.shape[1] - samples_short):
    #                 divs[i] = cp.shape[0]
    #             else:
    #                 divs[i] = cp.shape[0] - 1
    #         single_period = np.sum(cp, 0) / divs  # get the mean manually

    #     projection = np.tile(single_period, int(data.size / p) + 1)[
    #         : len(data)
    #     ]  # extend the period and take the good part

    #     # a faster, cleaner way to orthogonalize that is equivalent to the method
    #     # presented in "Orthogonal, exactly periodic subspace decomposition" (D.D.
    #     # Muresan, T.W. Parks), 2003. Setting trunc_to_integer_multiple gives a result
    #     # that is almost exactly identical (within a rounding error; i.e. 1e-6).
    #     # For the outputs of each to be identical, the input MUST be the same length
    #     # with DC removed since the algorithm in Muresan truncates internally and
    #     # here we allow the output to assume the dimensions of the input. See above
    #     # line of code.
    #     if orthogonalize:
    #         for f in get_factors(p, remove_1_and_n=True):
    #             if f in QOPeriods.PRIMES:
    #                 # remove the projection at p/prime_factor, taking care not to remove things twice.
    #                 projection = projection - QOPeriods.project(
    #                     projection, int(p / f), trunc_to_integer_multiple, False
    #                 )

    #     if return_single_period:
    #         return projection[0:p]  # just a single period
    #     else:
    #         return projection  # the whole thing

    # @staticmethod
    # def periodic_norm(x, p=None):
    #     """
    #     Calculate the periodic norm of a vector.

    #     Parameters
    #     ----------
    #     x : array_like
    #         Input vector.
    #     p : int, optional
    #         Period of the signal. If specified, the result is normalized by `sqrt(p)`.

    #     Returns
    #     -------
    #     float
    #         The periodic norm of `x`. If `p` is specified, the result is normalized by `sqrt(p)`.
    #     """
    #     if p:
    #         return (np.linalg.norm(x) / np.sqrt(len(x))) / np.sqrt(p)
    #     else:
    #         return np.linalg.norm(x) / np.sqrt(len(x))

    ################################################################################
    ### Detection ##################################################################
    ################################################################################
    def find_periods(
        self,
        data,
        num=None,
        thresh=None,
        min_length=2,
        max_length=None,
        update_weights=True,
        **kwargs,  # test_function=None, thresh=None
    ):
        """
        Use quadratic optimization to find the best periods in a signal.

        This function uses quadratic optimization to find the best periods in a signal. It starts by initializing a set of possible periods and iteratively finds the period that best represents the current residual of the signal. The residual is updated by subtracting the reconstruction of the signal using the found periods.

        The function can stop the iterative process based on a threshold on the root mean square of the residual or a custom test function. It can also control whether to update the weights for the existing periods when a new period is found.

        Parameters
        ----------
        data : np.ndarray
            The input signal as a 1-D array.
        num : int, optional
            The maximum number of periods to find. If not provided, it will be set to the length of the signal.
        thresh : float, optional
            The threshold on the root mean square of the residual to stop the iterative process. It is used when no custom test function is provided.
        min_length : int, optional
            The minimum length of the periods to consider. Default is 2.
        max_length : int, optional
            The maximum length of the periods to consider. If not provided, it will be set to one third of the length of the signal.
        update_weights : bool, optional
            Whether to update the weights for the existing periods when a new period is found. Default is True.
        **kwargs : dict, optional
            Additional keyword arguments. Can include a custom test function to control when to stop the iterative process.

        Returns
        -------
        tuple
            - output_bases (dict): A dictionary containing the found periods, their norms, the basis subspaces used to represent the signal, the weights for the basis subspaces, and a dictionary describing the construction of the basis subspaces.
            - res (np.ndarray): The final residual of the signal after subtracting the reconstruction using the found periods.

        Notes
        -----
        In each iteration, the function uses the method of orthogonalizing the projection to find the best period. This involves projecting the residual onto the subspace spanned by the period and then orthogonalizing the projection. The period that gives the maximum norm of the orthogonalized projection is chosen as the best period. This process repeats until the number of periods equals `num` or the test function returns `False`.

        The function also handles the case where the signal is virtually empty (i.e., its absolute sum is less than or equal to 1e-16). In this case, it returns a dictionary with period 1, norm 0, a subspace of ones, and weight 0, and a residual of zeros.

        If a singular matrix is encountered when solving the quadratic optimization problem, the function goes back one iteration and stops the process.
        """

        """
        Use quadradic optimization to find the best periods in a signal.

        The biggest trick is finding a good min_length (smallest acceptable period that is not a GCD) that describes the data well and does not introduce small periods (high frequencies) in the found periods. The most promising method so far (for sound, at least) is to take the spectral centroid of the data, weighting low frequencies by dividing the magnitudes by the bin number and use the result as the smallest allowable period. (SOLVED??? 09/08/2021).

        To Do:
            - instead of passing a threshold, should pass a function that is passed
                the residual. That way, arbitrary operations can be used to stop
                processing (i.e. spectral charactaristics)
        """

        N = len(data)
        if max_length is None:
            max_length = int(np.floor(N / 3))
        if num is None:
            num = N  # way too many but whatever
        periods = np.zeros(num, dtype=np.uint32)  # initialize
        possible_periods = np.arange(min_length, max_length + 1)
        norms = np.zeros(num)  # initialize
        bases = np.zeros((num, N))  # initialize
        res = data.copy()  # copy
        output_weights = np.array([])
        basis_matricies = np.empty((0, N))
        basis_dictionary = {}

        # if test_function is set, thresh is overridden
        if "test_function" in kwargs.keys():
            test_function = kwargs["test_function"]
        else:
            test_function = lambda self, x, y: rms(y) > (rms(data) * thresh)

        ## default to what we'd get if it were a vector of 0's ##
        if np.sum(np.abs(data)) <= 1e-16:
            output_bases = {
                "periods": np.array([1]),
                "norms": np.array([0]),
                "subspaces": np.ones((1, len(data))),
                "weights": np.array([0]),
                "basis_dictionary": {"1": len(data)},
            }
            self._output = output_bases
            return (
                output_bases,
                np.zeros(N),
            )  # just return a vector of 0's since the signal is virtually empty
        else:
            # these get recomputed (!!!) each time but stick them here so we can exit whenever
            output_bases = {
                "periods": [],
                "norms": [],
                "subspaces": [],
                "weights": [],
                "basis_dictionary": {},
            }

        for i in range(num):  # loop over the number of periods to find
            if i == 0 or test_function(
                self, data, reconstruction
            ):  # if the test function returns True (i.e. continue)
                if self._verbose:
                    print(
                        f"##########################################\ni = {i}\n##########################################"
                    )
                best_p = 0
                best_norm = 0
                best_base = None

                if self._orthogonalize:
                    # find the best period by orthogonalizing the projection

                    ## the choice to update weights depends on whether or not we want to allow the algorithm to alter
                    ## the previously computed weighted for the periods. This is the standard procedure but can result in
                    ## nonsense periods or the distortion of periods that are reasonable.
                    if update_weights:
                        best_p = self.get_best_period_orthogonal(
                            res, max_length, normalize=True
                        )
                        if self._verbose:
                            print("Best period here: {}".format(best_p))
                        if best_p < 1:
                            break  # stop if the best period is 0
                        # best_base = self.project(
                        #     data=res, p=best_p, trunc_to_integer_multiple=self._trunc_to_integer_multiple, orthogonalize=True
                        # ) # project out best_p from res, orthogonalize it, and return it
                        best_norm = self.periodic_norm(
                            best_base, best_p
                        )  # get the norm of the orthogonalized projection
                    else:
                        powers = self.get_best_period_orthogonal(
                            res, max_length, normalize=True, return_powers=True
                        )
                        powers = (-powers).argsort()
                        for p in powers:
                            if p in periods:
                                pass
                            else:
                                best_p = p
                                break
                        if self._verbose:
                            print("Best period here: {}".format(best_p))
                        if best_p < 1:
                            break
                        # best_base = self.project(
                        #     res, best_p, self._trunc_to_integer_multiple, True
                        # )  # always orthogonalize
                        best_norm = self.periodic_norm(best_base, best_p)
                else:
                    # find the best period by just plain projected
                    for p in possible_periods:
                        this_base = self.project(
                            res, p, self._trunc_to_integer_multiple, False
                        )  # dont orthogonalize the projection
                        this_norm = self.periodic_norm(this_base, p)
                        if this_norm > best_norm:
                            best_p = p
                            best_norm = this_norm
                            # best_base = this_base

                # now that we've found the strongest period in this run, remember it, its norm, and the projection
                periods[i] = best_p
                norms[i] = best_norm
                # bases[i] = best_base
                if self._verbose:
                    print(f"New period: {best_p}")

                nonzero_periods = periods[periods > 0]  # periods that are not 0
                print(nonzero_periods)
                # tup = self.compute_reconstruction(data, nonzero_periods, window)
                #
                # # this means that a singular matrix was encountered. Stop here.
                # if tup is None:
                #     break
                # else:
                #     # otherwise continue
                #     reconstruction = tup[0]
                #     output_bases = tup[1]
                #
                #     res = data - reconstruction  # get the residual
                #     output_bases['norms'] = norms[:len(nonzero_periods)]
                #     self._output_bases = output_bases

                ##########################
                """
                Now we have all of the basis subspaces AND their factors, correctly shaped.
                Find the best representation of the signal using the non-zero periods and
                subtract that from the original to get the new residual.
                """
                ##########################
                try:
                    if update_weights:
                        (
                            basis_matricies,
                            basis_dictionary,
                            output_weights,
                            reconstruction,
                        ) = self._update_weights(
                            data, N, nonzero_periods
                        )  # solve the quadratic optimization problem
                        res = (
                            data - reconstruction
                        )  # get the residual by subtracting the reconstruction from the original
                    else:
                        (
                            basis_matricies,
                            basis_dictionary,
                            output_weights,
                            reconstruction,
                        ) = self._dont_update_weights(
                            res,
                            N,
                            nonzero_periods,
                            output_weights,
                            basis_matricies,
                            basis_dictionary,
                        )
                        res = res - reconstruction

                    if self._verbose:
                        print("\tDictionary: {}".format(basis_dictionary))

                    ## set things here since it's also passed to test_function() ##
                    output_bases = {
                        "periods": nonzero_periods,
                        "norms": norms[: len(nonzero_periods)],
                        "subspaces": basis_matricies,
                        "weights": output_weights,
                        "basis_dictionary": basis_dictionary,
                    }
                    self._output_bases = output_bases

                except (
                    np.linalg.LinAlgError
                ):  # in the event of a singular matrix, go back one and call it good
                    if self._verbose:
                        print(
                            "\tSingular matrix encountered: going back one iteration and exiting loop"
                        )
                    break
            else:  # test_function returned False. Exit.
                if update_weights:
                    (
                        basis_matricies,
                        basis_dictionary,
                        output_weights,
                        reconstruction,
                    ) = self._update_weights(data, N, nonzero_periods)
                else:
                    (
                        basis_matricies,
                        basis_dictionary,
                        output_weights,
                        reconstruction,
                    ) = self._dont_update_weights(
                        res,
                        N,
                        nonzero_periods,
                        output_weights,
                        basis_matricies,
                        basis_dictionary,
                    )

                ## set things here since it's also passed to test_function() ##
                output_bases = {
                    "periods": nonzero_periods[
                        :-1
                    ],  # don't include the last one since that's what broke the test function
                    "norms": norms[: len(nonzero_periods) - 1],  # ibid
                    "subspaces": basis_matricies,  # include all since we didn't add to it when testing
                    "weights": output_weights,  # include all since we didn't add to it when testing
                    "basis_dictionary": basis_dictionary,  # include all since we didn't add to it when testing
                }
                self._output_bases = output_bases  # remember the output bases
                break  # test_function returned False. Exit.

        return (output_bases, res)

    def _update_weights(
        self, data: np.ndarray, N: int, nonzero_periods: np.ndarray
    ) -> tuple:
        """
        Update the weights for the given data and periods.

        This function takes the input data, the number of data points, and a list of 
        non-zero periods, and computes the optimal weights for representing the data 
        using the basis subspaces corresponding to the periods. It returns the basis 
        matrices, a dictionary describing the structure of the basis matrices, the 
        weights, and the reconstruction of the data using these weights.

        Parameters
        ----------
        data : np.ndarray
            The input data as a 1-D array.
        N : int
            The number of data points.
        nonzero_periods : np.ndarray
            A 1-D array of non-zero periods.

        Returns
        -------
        tuple
            - basis_matrices (np.ndarray): A 2-D array where each row is a basis 
            subspace corresponding to a period.
            - basis_dictionary (dict): A dictionary with periods as keys and the 
            number of rows for each period as values.
            - output_weights (np.ndarray): A 1-D array of weights for representing 
            the data using the basis subspaces.
            - reconstruction (np.ndarray): A 1-D array of the reconstruction of the 
            data using the weights and basis subspaces.
        """
        basis_matricies, basis_dictionary = self.get_subspaces(
            nonzero_periods, N
        )  # get the subspaces and a dictionary describing its construction (period, rows)
        """
        Now we have all of the basis subspaces AND their factors, correctly shaped.
        Find the best representation of the signal using the non-zero periods and
        subtract that from the original to get the new residual.
        """

        output_weights, reconstruction = self.solve_quadratic(
            data, basis_matricies, window=self.window
        )  # get the new reconstruction and do it again
        return (basis_matricies, basis_dictionary, output_weights, reconstruction)

    def _dont_update_weights(
        self,
        data: np.ndarray,
        N: int,
        nonzero_periods: np.ndarray,
        output_weights: np.ndarray,
        basis_matricies: np.ndarray,
        basis_dictionary: dict,
    ) -> tuple:
        """
        Find the weights for the given data while adding `nonzero_periods[-1]` without updating existing weights.

        This function is similar to the `_update_weights` method, but it does not update 
        the weights for the existing periods. Instead, it only computes the weights for the 
        new period (the last one in `nonzero_periods`). It then updates the basis matrices, 
        the basis dictionary, and the weights with the new period's information.

        Parameters
        ----------
        data : np.ndarray
            The input data as a 1-D array.
        N : int
            The number of data points.
        nonzero_periods : np.ndarray
            A 1-D array of non-zero periods.
        output_weights : np.ndarray
            A 1-D array of existing weights.
        basis_matricies : np.ndarray
            A 2-D array where each row is a basis subspace corresponding to a period.
        basis_dictionary : dict
            A dictionary with periods as keys and the number of rows for each period as values.

        Returns
        -------
        tuple
            - basis_matrices (np.ndarray): The updated 2-D array where each row is a basis 
            subspace corresponding to a period.
            - basis_dictionary (dict): The updated dictionary with periods as keys and the 
            number of rows for each period as values.
            - output_weights (np.ndarray): The updated 1-D array of weights for representing 
            the data using the basis subspaces.
            - reconstruction (np.ndarray): A 1-D array of the reconstruction of the 
            data using the updated weights and basis subspaces.
        """
        keep = nonzero_periods[-1]
        factors = get_factors(nonzero_periods[-1], remove_1=False, remove_n=False)
        factors_existing = set()
        for p in nonzero_periods[:-1]:
            print(p)
            factors_existing = factors_existing.union(
                get_factors(p, remove_1=False, remove_n=False)
            )

        for f in sorted(list(factors_existing.intersection(factors))):
            keep -= phi(f)
            print(
                "period: {}, keep: {}, thisFactor: {}".format(
                    nonzero_periods[-1], keep, f
                )
            )
        basis_matrix = self.Pp(nonzero_periods[-1], N, keep=keep)

        weights, reconstruction = self.solve_quadratic(
            data, basis_matrix, window=self.window
        )
        basis_dictionary.update({str(nonzero_periods[-1]): keep})
        basis_matricies = np.vstack((basis_matricies, basis_matrix))
        output_weights = np.concatenate((output_weights, weights))

        return (basis_matricies, basis_dictionary, output_weights, reconstruction)
    
    ################################################################################
    ### Extraction #################################################################
    ################################################################################
    def get_periods(self, weights, dictionary, decomp_type="row reduction"):
        periods = np.array([int(p) for p in dictionary.keys()])  # get the periods
        concatenated_periods = self.concatenate_periods(weights, dictionary)
        A = self.stack_pairwise_gcd_subspaces(periods)

        if decomp_type == "row reduction":
            A = reduce_rows(A)  # ought not to introduce truncation erros with integers
            coeffs, reconstructed = self.solve_quadratic(concatenated_periods, A, self._k, type="solve")
        elif decomp_type == "lu":
            PL, U = spla.lu(A, permute_l=True)
            coeffs, reconstructed = self.solve_quadratic(concatenated_periods, U, self._k, type="solve")
        elif decomp_type == "qr":
            Q, R = np.linalg.qr(A, mode="complete")
            coeffs, reconstructed = self.solve_quadratic(concatenated_periods, R, self._k, type="solve")
        else:
            coeffs, reconstructed = self.solve_quadratic(concatenated_periods, A, self._k, type="lstsq")

        actual_concatenated_periods = concatenated_periods - reconstructed
        actual_periods = []
        for i, p in enumerate(periods):
            start = int(np.sum(periods[:i]))
            actual_periods.append(actual_concatenated_periods[start : start + p])
        return tuple(actual_periods)

    @staticmethod
    def solve_quadratic(
        x: np.ndarray,
        A: np.ndarray,
        type: str = "solve",
        window: np.ndarray = None,
        k: int = 0,
    ):
        """
        Solve the quadratic equation Ax = b, where A is the 'basis matrix' and x is the data. The method supports both direct solving and least squares solving.

        Parameters
        ----------
        x : np.ndarray
            The 1D data vector.
        A : np.ndarray
            The 'basis matrix'.
        type : str, optional
            The type of solving method to use. Options are 'solve' for direct solving and 'lstsq' for least squares solving. Default is 'solve'.
        window : np.ndarray, optional
            The windowing function to apply to the data. If None or False, no windowing is applied. Default is None.
        k : int, optional
            The regularization parameter. Default is 0.

        Returns
        -------
        tuple
            output : np.ndarray
                The solution to the equation Ax = b.
            reconstructed : np.ndarray
                The reconstructed data from the solution.

        Notes
        -----
        The method first calculates the covariance matrix A' = A*A^T and the vector W = Ax. It then solves the equation A'x = W using the specified method. If the method is not recognized, it defaults to least squares solving.
        """
        #### windowing
        if (window is None) or (window is False):
            A_prime = np.matmul(A, A.T)  # multiply by its transpose (covariance)
            W = np.matmul(A, x)  # multiply by the data
        else:
            A_prime = np.matmul(
                A * window, A.T
            )  # multiply by its transpose (covariance)
            W = np.matmul(A * window, x)  # multiply by the data
        ####

        ## Regularization ## if doing this, don't drop rows from A, keep them all
        # A_prime = A_prime + (k * np.sum(np.power(x, 2)) * np.identity(A_prime.shape[0]))

        if type == "solve":
            output = np.linalg.solve(A_prime, W)  # actually solve it
            reconstructed = np.matmul(A.T, output)  # reconstruct the output
            return (output, reconstructed)
        elif type == "lstsq":
            output = np.linalg.lstsq(A_prime, W, rcond=None)  # actually solve it
            reconstructed = np.matmul(A.T, output[0])  # reconstruct the output
            return (output[0], reconstructed)
        else:
            warnings.warn("type ({}) unrecognized. Defaulting to lstsq.".format(type))
            output = np.linalg.lstsq(A_prime, W, rcond=None)  # actually solve it
            reconstructed = np.matmul(A.T, output[0])  # reconstruct the output
            return (output[0], reconstructed)

    def get_subspaces(self, Q: set, N: int) -> tuple:
        """
        Get the stacked subspaces for all periods in Q. This method only keeps the rows required to reconstruct the signal and nothing more.

        Parameters
        ----------
        Q : set
            The set of periods for which to get the subspaces.
        N : int
            The dimension of the subspace.

        Returns
        -------
        tuple
            A : ndarray
                The 'basis matrix' that is passed to solve_quadratic.
            d : dict
                A dictionary that holds information about A where the key-value pair is (q, rows) where q is the period and rows is the number of rows retained.

        Notes
        -----
        The method calculates the Euler's totient of all factors in R for each period in Q and stacks the matrices accordingly.
        """
        old_dimensionality = 0
        d = {}
        R = set()  # an empty set that is unioned with every new set of factors
        for q in Q:
            F = get_factors(q)  # get all the factors of q
            R = R.union(F)  # union of all previous factors with new factors
            s = np.sum(
                [phi(r) for r in R]
            )  # sum the Eulers totient of all factors in R
            d[str(q)] = s - old_dimensionality  # get the dimensionality of this q
            old_dimensionality = s  # remember the old dimensionality

        ## stack matricies as necessary
        A = np.array([]).reshape((0, N))
        if self._k == 0:
            for q, keep in d.items():
                # A = np.vstack((A, self.Pp(int(q), N, keep, self._basis_type) * np.sqrt(int(q)) ))
                A = np.vstack((A, self.Pp(int(q), N, keep, self._basis_type)))
        else:
            for q, keep in d.items():
                # A = np.vstack((A, self.Pp(int(q), N, None, self._basis_type)))
                A = np.vstack((A, self.Pp(int(q), N, keep, self._basis_type)))
        return (A, d)

    @staticmethod
    def concatenate_periods(weights:list, dictionary:dict):
        """
        Concatenate periods into a single vector.

        This function takes a list of weights and a dictionary that describes the 
        structure of the weights and concatenates them into a single vector. The 
        dictionary should have periods as keys and the number of weights for each 
        period as values.

        Parameters
        ----------
        weights : list
            List of weights. The order of the weights should correspond to the 
            order of the periods in the dictionary.
        dictionary : dict
            Dictionary with periods as keys and the number of weights for each 
            period as values.

        Returns
        -------
        np.ndarray
            A 1-D array of weights, concatenated according to the structure 
            described by the dictionary.
        """
        read_idx = 0
        output = []
        for q, r in dictionary.items():
            v = np.zeros(int(q))  # make a zero vector
            v[0:r] = weights[read_idx : read_idx + r]  # set the weights that we have
            read_idx += r  # incremenet our read index
            output.append(v)
        output = flatten(output)
        return np.array(output)

    def stack_pairwise_gcd_subspaces(self, periods):
        """
        Stack pairwise greatest common divisor (GCD) subspaces.

        This function takes a list of periods and creates a matrix where each row 
        is a subspace that corresponds to the GCD of a pair of periods. The subspaces 
        are created by taking the GCD of each pair of periods, creating a vector of 
        length equal to the period with the GCD at the corresponding indices, and 
        then stacking these vectors to form the matrix.

        Parameters
        ----------
        periods : list
            List of periods.

        Returns
        -------
        np.ndarray
            A 2-D array where each row is a subspace that corresponds to the GCD 
            of a pair of periods. The number of rows is equal to the number of 
            pairs of periods, and the number of columns is equal to the sum of the 
            periods.
        """
        subspace = []
        all_pairs = list(itertools.combinations(periods, 2))
        if len(periods) > 1:
            for pair in all_pairs:
                gcd = np.gcd(pair[0], pair[1])
                row = np.array([], dtype=np.int64)
                for p in periods:
                    if p in set(pair):
                        if p == pair[0]:
                            ss = self.Pp_column(gcd, 0, int(p / gcd)) * -1
                        else:
                            ss = self.Pp_column(gcd, 0, int(p / gcd))
                    else:
                        ss = np.zeros(p)

                    # ss /= int(p/gcd)
                    row = np.append(row, ss)

                subspace.append(row)  # essentiall np.roll(row, 0)
                for i in range(1, gcd):
                    subspace.append(np.roll(row, i))

            return np.vstack(tuple(subspace))
        elif len(periods) == 1:
            return np.ones((1, periods[0]))  # no need to redistribute
        else:
            return np.ones((1, 1))  # no need to redistribute

    @staticmethod
    def Pp(p:int, N:int=1, keep:int=None, type:str="natural") -> np.ndarray:
        """
        Generate a matrix of subspaces using the specified basis vector.

        Parameters
        ----------
        p : int
            The length of the basis vector.
        N : int, optional
            The number of columns in the output matrix. Default is 1.
        keep : int, optional
            The number of rows to keep in the output matrix. If specified, the output matrix is truncated to this number of rows. Default is None, in which case all rows are kept.
        type : str, optional
            The type of basis vector to use. Options are "natural" and "ramanujan". Default is "natural".

        Returns
        -------
        np.ndarray
            The generated matrix of subspaces.

        Notes
        -----
        The function first determines the number of times to repeat the basis vector based on the specified number of columns N. It then generates a matrix where each row is a basis vector. The type of basis vector used is determined by the 'type' parameter. If the 'keep' parameter is specified, the output matrix is truncated to this number of rows.
        """
        repetitions = int(np.ceil(N / p)) # (N // p) + 1
        matrix = np.zeros((p, int(N)))
        for i in range(p):
            if type == "natural":
                matrix[i] = QOPeriods.Pp_column(p, i, repetitions)[: int(N)]
            elif type == "ramanujan":
                matrix[i] = QOPeriods.Cq(p, i, repetitions, "real")[: int(N)]
        if keep:
            matrix = matrix[:keep]
        return matrix

    @staticmethod
    def Pp_column(p:int, s:int, repetitions:int=1):
        """
        Generate a natural basis periodic vector (i.e. [1, 0, 0, 0, 1, 0, 0, 0, 1, ...]])

        Parameters
        ----------
        p : int
            The period of the basis vector.
        s : int
            The shift parameter. The value 1 will be placed at indices where (index - s) is a multiple of p.
        repetitions : int, optional
            The number of times to repeat the basis vector. Default is 1.

        Returns
        -------
        np.ndarray
            The generated basis vector.

        Notes
        -----
        The function first creates a zero vector of length p. It then places a 1 at each index where (index - s) is a multiple of p. The resulting vector is then repeated the specified number of times.
        """
        vec = np.zeros(p)
        for i in range(p):
            if (i - s) % p == 0:
                vec[i] = 1
        return np.tile(vec, repetitions)

    @staticmethod
    def Cq(q:int, s:int, repetitions:int=1, type:str="real"):
        """
        Compute a Ramanujan periodic matrix.

        This function creates a Ramanujan periodic matrix, which is a matrix that has 
        the property that its rows are periodic with period q. The elements of the matrix 
        are complex numbers that are calculated using the Ramanujan sum, which is a 
        mathematical function that sums over the roots of unity.

        Parameters
        ----------
        q : int
            The period of the matrix. This is the number of unique rows in the matrix.
        s : int
            The shift of the matrix. This is the number of positions to roll the elements 
            of the matrix to the right.
        repetitions : int, optional
            The number of times to repeat the matrix. Default is 1.
        type : str, optional
            The type of the output. If "real", the real part of the complex numbers is returned. 
            If "complex", the complex numbers are returned as is. Default is "real".

        Returns
        -------
        np.ndarray
            The Ramanujan periodic matrix. It is a 1-D array of floats if type is "real", 
            and a 1-D array of complex numbers if type is "complex". The length of the array 
            is q * repetitions.
        """
        vec = np.zeros(q, dtype=complex)
        k = []
        for i in range(q):
            i += 1
            v = np.gcd(i, q)
            if v == 1:
                k.append(i)
        for i in range(q):
            for ii in k:
                vec[i] = vec[i] + np.exp(1j * 2 * np.pi * ii * i / q)

        vec = np.tile(np.roll(vec, s), repetitions)
        if type == "real":
            return np.real(vec)
        elif type == "complex":
            return vec
        else:
            return np.real(vec)

    def compute_reconstruction(self, x:np.ndarray, periods:list, type:str="lstsq", window:np.ndarray=None) -> tuple:
        """
        Compute the best representation of the signal using the non-zero periods and subtract that from the original to get the new residual.

        Parameters
        ----------
        x : np.ndarray
            The input signal.
        periods : list
            The list of periods to use for the reconstruction.
        type : str, optional
            The method to use for solving the quadratic equation. Options are "lstsq" and "solve". Default is "lstsq".
        window : np.ndarray, optional
            The window to apply to the signal before reconstruction. Default is None, in which case no window is applied.

        Returns
        -------
        tuple
            A tuple containing the reconstructed signal and a dictionary with information about the basis used for the reconstruction.

        Raises
        ------
        np.linalg.LinAlgError
            If a singular matrix is encountered during the reconstruction process.

        Notes
        -----
        The function first gets the subspaces and a dictionary describing its construction. It then tries to find the best representation of the signal using the non-zero periods and subtracts that from the original to get the new residual. If a singular matrix is encountered during this process, the function returns None.
        """
        basis_matricies, basis_dictionary = self.get_subspaces(
            periods, len(x)
        )  # get the subspaces and a dictionary describing its construction
        if self._verbose:
            print("\tDictionary: {}".format(basis_dictionary))

        ##########################
        """
        Now we have all of the basis subspaces AND their factors, correctly shaped.
        Find the best representation of the signal using the non-zero periods and
        subtract that from the original to get the new residual.
        """
        ##########################
        try:
            output_weights, reconstruction = self.solve_quadratic(
                x, basis_matricies, window=window, type=type
            )  # get the new reconstruction and do it again

            ## set things here since it's also passed to test_function() ##
            output_bases = {
                "periods": periods,
                "subspaces": basis_matricies,
                "weights": output_weights,
                "basis_dictionary": basis_dictionary
                # norms: defined outside
            }
        except np.linalg.LinAlgError:
            if self._verbose:
                print(
                    "\tSingular matrix encountered: going back one iteration and exiting loop"
                )
            return None

        return (reconstruction, output_bases)

    #######################################################
    ## Equations for orthogonal period finding. Straight from Muresan with slight
    ## optimization tweaks.
    #######################################################
    def eq_3(self, x, P):
        """
        Compute the value of Equation 3 from the Muresan paper.

        This equation is used to find the best period of a signal by calculating the 
        sum of the autocorrelations of the signal at lags that are multiples of the period.

        Parameters
        ----------
        x : np.ndarray
            The input signal, a 1-D array of floats.
        P : int
            The period to use in the calculation, an integer.

        Returns
        -------
        float
            The value of Equation 3 for the given signal and period. It represents the 
            sum of the autocorrelations of the signal at lags that are multiples of the period.
        """
        N = len(x)
        fac = P / N
        second_term = 0
        M = N // P
        for l in range(1, M):
            second_term += self.auto_corr(x, int(l * P))
        second_term *= 2
        return fac * (self.auto_corr(x, 0) + second_term)

    def auto_corr(self, x, k):
        """
        Compute the autocorrelation of the signal at a given lag. Note that this necessarily truncates the input signal. :(

        Autocorrelation is a mathematical tool for finding repeating patterns in a signal. 
        It is the correlation of the signal with itself at different points in time.

        Parameters
        ----------
        x : np.ndarray
            The input signal, a 1-D array of floats.
        k : int
            The lag to use in the autocorrelation calculation, an integer.

        Returns
        -------
        float
            The autocorrelation of the signal at lag k. It represents the degree of similarity 
            between the signal and its lagged version.
        """
        N = len(x)
        A = np.vstack((x[0 : N - k], x[k:N]))
        return np.sum(np.prod(A, 0))  # multiply column-wise, then sum

    def get_best_period_orthogonal(
        self, x, max_p=None, normalize=False, return_powers=False
    ):
        """
        Find the best period of a signal using Equation 3 of the Muresan paper.

        This method first calculates the autocorrelation for each possible period up to max_p. 
        It then subtracts the power of each factor of each period from the power of that period. 
        Finally, it returns the period with the maximum power, or the powers for all periods 
        if return_powers is True.

        Parameters
        ----------
        x : np.ndarray
            The input signal, a 1-D array of floats.
        max_p : int, optional
            The maximum period to consider. Default is half the length of the signal.
        normalize : bool, optional
            Whether to normalize the powers by the period. Default is False.
        return_powers : bool, optional
            Whether to return the powers for all periods instead of just the best period. Default is False.

        Returns
        -------
        int or np.ndarray
            If return_powers is False, returns the best period, i.e., the period that maximizes the 
            value of Equation 3. If return_powers is True, returns the powers for all periods, i.e., 
            the values of Equation 3 for all periods.
        """
        if max_p is None:
            max_p = (
                len(x) // 2
            )  # max period is half the length of the signal by default
        Q = np.arange(1, max_p)  # all possible periods
        pows = np.zeros(Q[-1] + 1)
        for q in Q:
            pows[q] = max(
                self.eq_3(x, q), 0
            )  # get the max of the autocorrelation for each q in Q
            facs = get_factors(q)  # get the factors of q
            for f in facs:  # for each factor
                if f != q:  # if it's not the period itself
                    pows[q] -= pows[f]  # subtract the power of that factor
        pows[
            pows < 0
        ] = 0  # set everything nevative to zero (kinda wonky, should revisit)

        if normalize:
            pows[1:] = pows[1:] / Q  # normalize by the period
        if return_powers:
            return pows
        else:
            if np.argmax(pows) > 0:
                return (
                    Q[np.argmax(pows)] - 1
                )  # return the strongest period, subtract one to offset the values in Q
            else:
                return 1  # if the max is 0, return 1 instead (same thing)

    #######################################################

    ### Properties
    def basis_type():
        doc = "The basis_type property."

        def fget(self):
            return self._basis_type

        def fset(self, value):
            self._basis_type = value

        return locals()

    basis_type = property(**basis_type())

    def verbose():
        doc = "The verbose property."

        def fget(self):
            return self._verbose

        def fset(self, value):
            self._verbose = value

        return locals()

    verbose = property(**verbose())

    def k():
        doc = "The k property for regularization"

        def fget(self):
            return self._k

        def fset(self, value):
            self._k = value

        return locals()

    k = property(**k())

    def orthogonalize():
        doc = "The orthogonalize property for regularization"

        def fget(self):
            return self._orthogonalize

        def fset(self, value):
            self._orthogonalize = value

        return locals()

    orthogonalize = property(**orthogonalize())

    def window():
        doc = "The window property for regularization"

        def fget(self):
            return self._window

        def fset(self, value):
            self._window = value

        return locals()

    window = property(**window())

    def output_bases():
        doc = "The output_bases property"

        def fget(self):
            return self._output_bases

        return locals()

    output_bases = property(**output_bases())
