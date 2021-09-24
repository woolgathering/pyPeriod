"""
    IMPORTANT!!
    You must use Python >= 3.6 for ordered dictionaries. Otherwise you are not
    guarenteed to get back the original periods from get_periods().
"""

import numpy as np
from functools import reduce
import itertools
import random
from scipy import linalg as spla
import warnings


def remove_random_rows(arr, keep):
    idxs = np.array(random.sample(range(arr.shape[0]), keep))
    return arr[idxs]


def phi(n):
    """
        Euler's totient function
    """
    amount = 0
    for k in range(1, n + 1):
        if np.gcd(n, k) == 1:
            amount += 1
    return amount


def get_factors(n, remove_1_and_n=False):
    """
        Get all factors of some n as a set
    """
    facs = set(
        reduce(list.__add__, ([i, n // i]
                              for i in range(1,
                                             int(n**0.5) + 1) if n % i == 0)))
    # get rid of 1 and the number itself
    if remove_1_and_n:
        facs.remove(1)
        facs.remove(n)
    return facs  # retuned as a set


def rms(x):
    return np.sqrt(np.sum(np.power(x, 2)) / len(x))


def flatten(t):
    return [item for sublist in t for item in sublist]


def reduce_rows(A):
    AA = A[0]
    rank = np.linalg.matrix_rank(AA)  # should be 1
    for row in A[1:]:
        aa = np.vstack((AA, row))
        if np.linalg.matrix_rank(aa) > int(rank):
            AA = aa
            rank = np.linalg.matrix_rank(aa)
    return AA


class QOPeriods:

    def __init__(self, basis_type='natural', trunc_to_integer_multiple=False):
        # super(Periods, self).__init__()
        self._trunc_to_integer_multiple = trunc_to_integer_multiple
        self._output = None
        self._basis_type = basis_type
        self._verbose = False

    @staticmethod
    def project(data,
                p=2,
                trunc_to_integer_multiple=False,
                return_single_period=False):
        cp = data.copy()
        samples_short = int(
            np.ceil(len(cp) / p) * p -
            len(cp))  # calc how many samples short for rectangle
        cp = np.pad(cp, (0, samples_short))  # pad it
        cp = cp.reshape(int(len(cp) / p), p)  # reshape it to a rectangle

        if trunc_to_integer_multiple:
            if samples_short == 0:
                single_period = np.mean(cp,
                                        0)  # don't need to omit the last row
            else:
                single_period = np.mean(
                    cp[:-1], 0
                )  # just take the mean of the truncated version and output a single period
        else:
            ## this is equivalent to the method presented in the paper but significantly faster ##
            # do the mean manually. get the divisors from the last row since the last samples_short values will be one less than the others
            divs = np.zeros(cp.shape[1])
            for i in range(cp.shape[1]):
                if i < (cp.shape[1] - samples_short):
                    divs[i] = cp.shape[0]
                else:
                    divs[i] = cp.shape[0] - 1
            single_period = np.sum(cp, 0) / divs  # get the mean manually

        if return_single_period:
            # return projection[0:p]  # just a single period
            return single_period
        else:
            projection = np.tile(
                single_period,
                int(data.size / p) +
                1)[:len(data)]  # extend the period and take the good part
            return projection  # the whole thing

    @staticmethod
    def periodic_norm(x, p=None):
        if p:
            return (np.linalg.norm(x) / np.sqrt(len(x))) / np.sqrt(p)
        else:
            return np.linalg.norm(x) / np.sqrt(len(x))

    ##############################################################################
    ##############################################################################
    ##############################################################################
    def find_periods(self,
                     data,
                     num=None,
                     thresh=None,
                     min_length=2,
                     max_length=None,
                     **kwargs):
        """
        Using quadradic optimization to find the best periods.

        The biggest trick is finding a good min_length (smallest acceptable period that
        is not a GCD) that describes the data well and does not introduce small periods
        (high frequencies) in the found periods. The most promising method so far (for
        sound, at least) is to take the spectral centroid of the data, weighting low
        frequencies by dividing the magnitudes by the bin number and use the result
        as the smallest allowable period. (SOLVED??? 09/08/2021)

        To Do:
            - instead of passing a threshold, should pass a function that is passed
                the residual. That way, arbitrary operations can be used to stop
                processing (i.e. spectral charactaristics)
        """

        N = len(data)
        if max_length is None:
            max_length = int(np.floor(N / 3))
        if num is None:
            num = N  # way too big
        periods = np.zeros(num, dtype=np.uint32)
        norms = np.zeros(num)
        bases = np.zeros((num, N))
        res = data.copy()  # copy

        # if test_function is set, thresh is overridden
        if 'test_function' in kwargs.keys():
            func = kwargs['test_function']
        else:
            test_function = lambda x: rms(x) > (rms(data) * thresh)

        # these get recomputed (!!!) each time but stick them here so we can exit whenever
        basis_matricies = None  # a list to later be changed into a tuple
        nonzero_periods = None  # periods that are not 0
        output_weights = None  # coefficients
        basis_dictionary = None  # dictionary describing the construction of basis_matricies

        for i in range(num):
            if self._verbose:
                print('Function returned: {}'.format(test_function(res)))
            if test_function(res):
                if self._verbose:
                    print('##########################################')
                best_p = 0
                best_norm = 0
                best_base = None
                for p in range(min_length, max_length + 1):
                    this_base = self.project(res, p,
                                             self._trunc_to_integer_multiple,
                                             False)
                    this_norm = self.periodic_norm(this_base, p)
                    if this_norm > best_norm:
                        best_p = p
                        best_norm = this_norm
                        best_base = this_base

                # now that we've found the strongest period in this run, set them
                periods[i] = best_p
                # norms[i] = best_norm
                # norms[i] = rms(best_base)
                # norms[i] = np.linalg.norm(best_base)
                norms[i] = np.sum(np.power(best_base, 2))
                bases[i] = best_base

                if self._verbose:
                    print('New period: {}'.format(best_p))

                basis_matricies = []  # a list to later be changed into a tuple
                nonzero_periods = periods[periods >
                                          0]  # periods that are not 0
                basis_matricies, basis_dictionary = self.get_subspaces(
                    nonzero_periods, N
                )  # get the subspaces and a dictionary describing its construction
                if self._verbose:
                    print('\tDictionary: {}'.format(basis_dictionary))

                ##########################
                """
                Now we have all of the basis subspaces AND their factors, correctly shaped.
                Find the best representation of the signal using the non-zero periods and
                subtract that from the original to get the new residual.
                """
                ##########################
                try:
                    resconst, output_weights = self.solve_quadratic(
                        data, basis_matricies
                    )  # get the new residual and do it again
                    res = data - resconst  # get the residual

                    if self._verbose:
                        print('\tWeights: {}'.format(output_weights))

                    # remember old stuff
                    old_basis = basis_matricies
                    old_nonzero_periods = nonzero_periods
                    old_basis_dictionary = basis_dictionary
                except np.linalg.LinAlgError:
                    # in the event of a singular matrix, go back one and call it
                    basis_matricies = old_basis
                    nonzero_periods = old_nonzero_periods
                    basis_dictionary = old_basis_dictionary

                    if self._verbose:
                        print(
                            '\tSingular matrix encountered: going back one iteration and exiting loop'
                        )
                    break
            else:
                break  # test_function returned False. Exit.

        output_bases = {
            'periods': nonzero_periods,
            'norms': norms[:len(nonzero_periods)],
            'subspaces': basis_matricies,
            'weights': output_weights,
            'basis_dictionary': basis_dictionary
        }
        self._output = output_bases
        return (output_bases, res)

    def get_periods(self, weights, dictionary, decomp_type='row reduction'):
        weights = self._output['weights']
        dictionary = self._output['basis_dictionary']
        periods = self._output['periods']
        cp = self.concatenate_periods(weights, dictionary)
        A = self.stack_pairwise_gcd_subspaces(periods)

        if decomp_type == 'row reduction':
            A = reduce_rows(A)
            reconst, coeffs = self.solve_quadratic(cp, A)
            actual_cp = cp - reconst
            actual_periods = []
            for i, p in enumerate(periods):
                start = int(np.sum(periods[:i]))
                actual_periods.append(actual_cp[start:start + p])
            return tuple(actual_periods)

        elif decomp_type == 'lu':
            pl, A = spla.lu(A, permute_l=True)
            A_prime = np.matmul(A, A.T)  # multiply by its transpose
            W = np.matmul(A, cp)  # multiply by the data
            result = np.linalg.lstsq(A_prime, W,
                                     rcond=None)  # actually solve it
            reconstructed = np.matmul(A.T, result[0])  # reconstruct the result
            actual_cp = cp - reconstructed
            actual_periods = []
            for i, p in enumerate(periods):
                start = int(np.sum(periods[:i]))
                actual_periods.append(actual_cp[start:start + p])
            return tuple(actual_periods)

        elif decomp_type == 'qr':
            q, A = spla.qr(A, pivoting=False)
            A_prime = np.matmul(A, A.T)  # multiply by its transpose
            W = np.matmul(A, cp)  # multiply by the data
            result = np.linalg.lstsq(A_prime, W,
                                     rcond=None)  # actually solve it
            reconstructed = np.matmul(A.T, result[0])  # reconstruct the result
            actual_cp = cp - reconstructed
            actual_periods = []
            for i, p in enumerate(periods):
                start = int(np.sum(periods[:i]))
                actual_periods.append(actual_cp[start:start + p])
            return tuple(actual_periods)

        else:
            raise Error('Unrecognized decomp_type: {}'.format(decomp_type))

    @staticmethod
    def solve_quadratic(x, A, type='solve'):
        A_prime = np.matmul(A, A.T)  # multiply by its transpose
        W = np.matmul(A, x)  # multiply by the data
        if type == 'solve':
            output = np.linalg.solve(A_prime, W)  # actually solve it
            reconstructed = np.matmul(A.T, output)  # reconstruct the output
            return (reconstructed, output)
        elif type == 'lstsq':
            output = np.linalg.lstsq(A_prime, W)  # actually solve it
            reconstructed = np.matmul(A.T, output[0])  # reconstruct the output
            return (reconstructed, output[0])
        else:
            warnings.warn(
                'type ({}) unrecognized. Defaulting to lstsq.'.format(type))
            output = np.linalg.lstsq(A_prime, W)  # actually solve it
            reconstructed = np.matmul(A.T, output[0])  # reconstruct the output
            return (reconstructed, output[0])

    def get_subspaces(self, Q, N):
        """
            Get the stacked subspaces for all periods in Q.
            This only keeps the rows required to reconstruct the signal and nothing more.

            Returns:
                A: the 'basis matrix' that is passed to solve_quadratic
                d: a dictionary that holds information about A where the key-value
                    pair is (q, rows) where q is the period and rows is the number
                    of rows retained.
        """
        old_dimensionality = 0
        d = {}
        R = set()  # an empty set that is unioned with every new set of factors
        for q in Q:
            F = get_factors(q)  # get all the factors of q
            R = R.union(F)  # union of all previous factors with new factors
            s = np.sum([phi(r) for r in R
                        ])  # sum the Eulers totient of all factors in R
            d[str(
                q
            )] = s - old_dimensionality  # get the dimensionality of this q
            old_dimensionality = s  # remember the old dimensionality

        ## stack matricies as necessary
        A = np.array([]).reshape((0, N))
        for q, keep in d.items():
            A = np.vstack(
                (A, self.Pt_complete(int(q), N, keep, self._basis_type)))
        return (A, d)

    @staticmethod
    def concatenate_periods(weights, dictionary):
        """
            Concatenate periods into a single vector in order to compute them.
        """
        read_idx = 0
        output = []
        for q, r in dictionary.items():
            v = np.zeros(int(q))  # make a zero vector
            v[0:r] = weights[read_idx:read_idx +
                             r]  # set the weights that we have
            read_idx += r  # incremenet our read index
            output.append(v)
        output = flatten(output)
        return np.array(output)

    def stack_pairwise_gcd_subspaces(self, periods):
        subspace = []
        all_pairs = list(itertools.combinations(periods, 2))
        for pair in all_pairs:
            gcd = np.gcd(pair[0], pair[1])
            row = np.array([], dtype=np.int64)
            for p in periods:
                if p in set(pair):
                    if p == pair[0]:
                        ss = self.Pp(gcd, 0, int(p / gcd)) * -1
                    else:
                        ss = self.Pp(gcd, 0, int(p / gcd))
                else:
                    ss = np.zeros(p)

                row = np.append(row, ss)

            subspace.append(row)  # essentiall np.roll(row, 0)
            for i in range(1, gcd):
                subspace.append(np.roll(row, i))

        return np.vstack(tuple(subspace))

    @staticmethod
    def Pt_complete(p, N=1, keep=None, type='natural'):
        """
            Make subspaces using the natural basis vector.

            Worth trying Ramanujan?!?!
        """
        repetitions = int(np.ceil(N / p))
        matrix = np.zeros((p, int(N)))
        for i in range(p):
            if type == 'natural':
                matrix[i] = QOPeriods.Pp(p, i, repetitions)[:int(N)]
            elif type == 'ramanujan':
                matrix[i] = QOPeriods.Cq(p, i, repetitions, 'real')[:int(N)]
        if keep:
            matrix = matrix[:keep]
            # matrix = remove_random_rows(matrix, keep)
        return matrix

    @staticmethod
    def Pp(p, s, repetitions=1):
        """
            Natural basis vector. Eg: [1, 0, 0, ...]
        """
        vec = np.zeros(p)
        for i in range(p):
            if (i - s) % p == 0:
                vec[i] = 1
        return np.tile(vec, repetitions)

    @staticmethod
    def Cq(q, s, repetitions=1, type='real'):
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
        if type == 'real':
            return np.real(vec)
        elif type == 'complex':
            return vec
        else:
            if self._verbose:
                print('Return type invalid, defaulting to \'real\'')
            return np.real(vec)

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
