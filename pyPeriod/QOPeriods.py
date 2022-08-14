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
        if n != 1:
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

def get_primes(max=1000000):
    primes = np.arange(3, max + 1, 2)
    isprime = np.ones((max - 1) // 2, dtype=bool)
    for factor in primes[:int(np.sqrt(max))]:
        if isprime[(factor - 2) // 2]:
            isprime[(factor * 3 - 2) // 2::factor] = 0
    return np.insert(primes[isprime], 0, 2)

def normalize(x, level=1):
    m = np.max(np.abs(x))
    return (x/m) * level

class QOPeriods:
    PRIMES = set(get_primes(10000))  # a class variable to hold a set of primes (does not include 1)

    def __init__(self, basis_type='natural', trunc_to_integer_multiple=False, orthogonalize=False):
        # super(Periods, self).__init__()
        self._trunc_to_integer_multiple = trunc_to_integer_multiple
        self._output = None
        self._basis_type = basis_type
        self._verbose = False
        self._k = 0
        self._orthogonalize = orthogonalize
        self._window = False

    @staticmethod
    def project(data,
                p=2,
                trunc_to_integer_multiple=False,
                orthogonalize = False,
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

        projection = np.tile(
            single_period,
            int(data.size / p) +
            1)[:len(data)]  # extend the period and take the good part

        # a faster, cleaner way to orthogonalize that is equivalent to the method
        # presented in "Orthogonal, exactly periodic subspace decomposition" (D.D.
        # Muresan, T.W. Parks), 2003. Setting trunc_to_integer_multiple gives a result
        # that is almost exactly identical (within a rounding error; i.e. 1e-6).
        # For the outputs of each to be identical, the input MUST be the same length
        # with DC removed since the algorithm in Muresan truncates internally and
        # here we allow the output to assume the dimensions of the input. See above
        # line of code.
        if orthogonalize:
            for f in get_factors(p, remove_1_and_n=True):
                if f in QOPeriods.PRIMES:
                    # remove the projection at p/prime_factor, taking care not to remove things twice.
                    projection = projection - QOPeriods.project(
                        projection, int(p / f), trunc_to_integer_multiple,
                        False)

        if return_single_period:
            return projection[0:p]  # just a single period
        else:
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
        old_res = np.zeros(len(data))
        if self._window:
            window = np.hanning(N)
        else:
            window = None

        # if test_function is set, thresh is overridden
        if 'test_function' in kwargs.keys():
            test_function = kwargs['test_function']
        else:
            test_function = lambda self, x, y, y1, o: rms(y) > (rms(data) * thresh)

        # these get recomputed (!!!) each time but stick them here so we can exit whenever
        basis_matricies = None  # a list to later be changed into a tuple
        nonzero_periods = None  # periods that are not 0
        output_weights = None  # coefficients
        basis_dictionary = None  # dictionary describing the construction of basis_matricies

        ## default to what we'd get if it were a vector of 0's ##
        if np.sum(np.abs(data)) <= 1e-16:
            output_bases = {
                'periods': np.array([1]),
                'norms': np.array([0]),
                'subspaces': np.ones((1, len(data))),
                'weights': np.array([0]),
                'basis_dictionary': {'1': len(data)}
            }
            self._output = output_bases
            return (output_bases, np.zeros(N))

        for i in range(num):
            if i == 0 or test_function(self, data, res, old_res, output_bases):
                if self._verbose:
                    print('##########################################\ni = {}'.format(i))
                best_p = 0
                best_norm = 0
                best_base = None

                if self._orthogonalize:
                    best_p = self.get_best_period_orthogonal(res, max_length, normalize=True)
                    best_base = self.project(res, best_p, self._trunc_to_integer_multiple, True) # always orthogonalize
                    best_norm = self.periodic_norm(best_base, best_p)
                else:
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
                norms[i] = best_norm
                bases[i] = best_base


                if self._verbose:
                    print('New period: {}'.format(best_p))

                nonzero_periods = periods[periods > 0]  # periods that are not 0
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
                    output_weights, reconstruction = self.solve_quadratic(
                        data, basis_matricies, window=window
                    )  # get the new reconstruction and do it again
                    old_res = res  # get the residual
                    res = data - reconstruction  # get the residual

                    ## set things here since it's also passed to test_function() ##
                    output_bases = {
                        'periods': nonzero_periods,
                        'norms': norms[:len(nonzero_periods)],
                        'subspaces': basis_matricies,
                        'weights': output_weights,
                        'basis_dictionary': basis_dictionary
                    }

                    # remember old stuff
                    old_basis = basis_matricies
                    old_nonzero_periods = nonzero_periods
                    old_basis_dictionary = basis_dictionary
                    old_output_weights = output_weights
                except np.linalg.LinAlgError:
                    # in the event of a singular matrix, go back one and call it
                    if self._verbose:
                        print(
                            '\tSingular matrix encountered: going back one iteration and exiting loop'
                        )
                    break
            else:
                break  # test_function returned False. Exit.

        self._output = output_bases
        return (output_bases, res)

    def get_periods(self, weights, dictionary, decomp_type='row reduction'):
        periods = np.array([int(p) for p in dictionary.keys()])
        cp = self.concatenate_periods(weights, dictionary)
        A = self.stack_pairwise_gcd_subspaces(periods)

        if decomp_type == 'row reduction':
            A = reduce_rows(A) # ought not to introduce truncation erros with integers
            coeffs, reconstructed = self.solve_quadratic(cp, A, self._k, type='solve')

        elif decomp_type == 'lu':
            PL, U = spla.lu(A, permute_l=True)
            coeffs, reconstructed = self.solve_quadratic(cp, U, self._k, type='solve')

        elif decomp_type == 'qr':
            # Q, R = spla.qr(A, pivoting=False)
            Q, R = np.linalg.qr(A)
            coeffs, reconstructed = self.solve_quadratic(cp, R, self._k, type='solve')

        else:
            # raise TypeError('Unrecognized decomp_type: {}'.format(decomp_type))
            coeffs, reconstructed = self.solve_quadratic(cp, A, self._k, type='lstsq')

        actual_cp = cp - reconstructed
        actual_periods = []
        for i, p in enumerate(periods):
            start = int(np.sum(periods[:i]))
            actual_periods.append(actual_cp[start:start + p])
        return tuple(actual_periods)

    @staticmethod
    def solve_quadratic(x, A, k=0, type='solve', window=None):

        #### windowing
        if window is not None:
            A_prime = np.matmul(A * window, A.T)  # multiply by its transpose (covariance)
            W = np.matmul(A * window, x)  # multiply by the data
        else:
            A_prime = np.matmul(A, A.T)  # multiply by its transpose (covariance)
            W = np.matmul(A, x)  # multiply by the data
        ####

        ## Regularization ## if doing this, don't drop rows from A, keep them all
        # A_prime = A_prime + (k * np.sum(np.power(x, 2)) * np.identity(A_prime.shape[0]))

        if type == 'solve':
            output = np.linalg.solve(A_prime, W)  # actually solve it
            reconstructed = np.matmul(A.T, output)  # reconstruct the output
            return (output, reconstructed)
        elif type == 'lstsq':
            output = np.linalg.lstsq(A_prime, W, rcond=None)  # actually solve it
            reconstructed = np.matmul(A.T, output)  # reconstruct the output
            return (output[0], reconstructed)
        else:
            warnings.warn(
                'type ({}) unrecognized. Defaulting to lstsq.'.format(type))
            output = np.linalg.lstsq(A_prime, W, rcond=None)  # actually solve it
            reconstructed = np.matmul(A.T, output)  # reconstruct the output
            return (output[0], reconstructed)

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
            s = np.sum([phi(r) for r in R])  # sum the Eulers totient of all factors in R
            d[str(q)] = s - old_dimensionality  # get the dimensionality of this q
            old_dimensionality = s  # remember the old dimensionality

        ## stack matricies as necessary
        A = np.array([]).reshape((0, N))
        if self._k == 0:
            for q, keep in d.items():
                A = np.vstack((A, self.Pp(int(q), N, keep, self._basis_type)))
        else:
            for q, keep in d.items():
                # A = np.vstack((A, self.Pp(int(q), N, None, self._basis_type)))
                A = np.vstack((A, self.Pp(int(q), N, keep, self._basis_type)))
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
            v[0:r] = weights[read_idx:read_idx + r]  # set the weights that we have
            read_idx += r  # incremenet our read index
            output.append(v)
        output = flatten(output)
        return np.array(output)

    def stack_pairwise_gcd_subspaces(self, periods):
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
                    row = np.append(row, ss)

                subspace.append(row)  # essentiall np.roll(row, 0)
                for i in range(1, gcd):
                    subspace.append(np.roll(row, i))

            return np.vstack(tuple(subspace))
        elif len(periods) == 1:
            return np.ones((1, periods[0])) # no need to redistribute
        else:
            return np.ones((1, 1)) # no need to redistribute

    @staticmethod
    def Pp(p, N=1, keep=None, type='natural'):
        """
            Make subspaces using the natural basis vector.
        """
        repetitions = int(np.ceil(N / p))
        matrix = np.zeros((p, int(N)))
        for i in range(p):
            if type == 'natural':
                matrix[i] = QOPeriods.Pp_column(p, i, repetitions)[:int(N)]
            elif type == 'ramanujan':
                matrix[i] = QOPeriods.Cq(p, i, repetitions, 'real')[:int(N)]
        if keep:
            matrix = matrix[:keep]
        return matrix

    @staticmethod
    def Pp_column(p, s, repetitions=1):
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

    #######################################################
    ## Equations for orthogonal period finding. Straight from Muresan with slight
    ## optimization tweaks.
    #######################################################
    def eq_3(self, x, P):
        N = len(x)
        fac = P/N
        second_term = 0
        M = N//P
        for l in range(1, M):
            second_term += self.auto_corr(x, int(l*P))
        second_term *= 2
        return fac * second_term

    def auto_corr(self, x, k):
        """
            Note that this necessarily truncates the input signal. :(
        """
        N = len(x)
        A = np.vstack((x[0:N-k], x[k:N]))
        return np.sum(np.prod(A, 0)) # multiply column-wise, then sum

    def get_best_period_orthogonal(self, x, max_p=None, normalize=False, return_powers=False):
        if max_p is None:
            max_p = len(x)//2
        Q = np.arange(1, max_p)
        pows = np.zeros(Q[-1]+1)
        for q in Q:
            pows[q] = max(self.eq_3(x, q), 0)
            facs = get_factors(q)
            for f in facs:
                if f != q:
                    pows[q] -= pows[f]
        pows[pows < 0] = 0 # set everything nevative to zero

        if normalize:
            pows[1:] =  pows[1:] / Q
        if return_powers:
            return pows
        else:
            return np.argmax(pows) # return the strongest period
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
