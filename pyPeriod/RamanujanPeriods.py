import numpy as np
from functools import reduce
import itertools
from numba import jit, vectorize, float64, float32, int64, int32

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
        reduce(list.__add__,
               ([i, n // i] for i in range(1,
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


class RamanujanPeriods:

    def __init__(self, basis_type='natural'):
        self._basis_type = basis_type
        self._output = None
        self._verbose = None


    def find_periods(self, x, min_length=2, max_length=None, select_periods=None):
        if not max_length:
            max_length = len(x) // 3

        norms = np.zeros(max_length+1)
        for p in range(min_length, max_length+1):
            basis = self.Cq_complete(p, len(x))
            projection = RamanujanPeriods.project(x, basis)
            output = np.sum(projection, 0)
            norms[p] = np.sum(np.power(output, 2))

        # norms = norms / np.abs(norms) # normalize it
        if select_periods:
            if hasattr(select_periods, '__call__'):
                return select_periods(norms)
        else:
            return norms

    def find_periods_with_weights(self, x, min_length=2, max_length=None, thresh=0.2):
        norms = self.find_periods(x, min_length, max_length, select_periods=None)
        norms = norms / np.abs(np.max(norms))
        periods = np.argwhere(norms > thresh).flatten()
        if self._verbose:
            print ('Found periods {}'.format(periods))

        basis_matricies, basis_dictionary = self.get_subspaces(
            periods, len(x)
        )  # get the subspaces and a dictionary describing its construction
        resconst, output_weights = self.solve_quadratic(
            x, basis_matricies
        )  # get the new residual and do it again
        res = x - resconst  # get the residual

        output_bases = {
            'periods': periods,
            'norms': norms[periods],
            'subspaces': basis_matricies,
            'weights': output_weights,
            'basis_dictionary': basis_dictionary
        }
        self._output = output_bases
        return (output_bases, res)

    def get_periods(self, weights, dictionary):
        weights = self._output['weights']
        dictionary = self._output['basis_dictionary']
        periods = self._output['periods']
        cp = self.concatenate_periods(weights, dictionary)
        A = self.stack_pairwise_gcd_subspaces(periods)
        print ('Dimensions of A_redistribution: {}'.format(A.shape))
        A = reduce_rows(A)
        reconst, coeffs = self.solve_quadratic(cp, A)
        actual_cp = cp - reconst
        actual_periods = []
        for i, p in enumerate(periods):
            start = int(np.sum(periods[:i]))
            actual_periods.append(actual_cp[start:start + p])
        return tuple(actual_periods)

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
                        # ss = self.Cq(gcd, 0, int(p / gcd)) * -1
                    else:
                        ss = self.Pp(gcd, 0, int(p / gcd))
                        # ss = self.Cq(gcd, 0, int(p / gcd))
                else:
                    ss = np.zeros(p)

                row = np.append(row, ss)

            subspace.append(row) # essentiall np.roll(row, 0)
            for i in range(1, gcd):
                subspace.append(np.roll(row, i))

        return np.vstack(tuple(subspace))

    @staticmethod
    @jit(nopython=False, parallel=True)
    def project(x, basis):
        proj_complete = np.zeros(basis.shape)
        for i,row in enumerate(basis):
            row = row / np.max(row)
            proj = np.dot(x, row) * row
            proj_complete[i] = proj
        return proj_complete

    # @staticmethod
    # @jit(nopython=True, parallel=True)
    # # @vectorize([int32(int32, int32), int64(int64, int64), float32(float32, float32), float64(float64, float64)])
    # def project(x, basis):
    #     # proj_complete = np.zeros(basis.shape)
    #     basis = basis / np.max(basis)
    #     # proj = np.matmul(np.tile(x, (basis.shape[0],1)).T, basis) * basis
    #     xx = x.repeat(basis.shape[0]).reshape((-1, basis.shape[0]))
    #     proj = np.sum(xx * basis, 1) * basis
    #     return proj

    @staticmethod
    # @jit(nopython=True, parallel=True)
    def solve_quadratic(x, A):
        A_prime = np.matmul(A, A.T)  # multiply by its transpose
        W = np.matmul(A, x)  # multiply by the data
        output = np.linalg.solve(A_prime, W)  # actually solve it
        reconstructed = np.matmul(A.T, output)  # reconstruct the output
        return (reconstructed, output)

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
                q)] = s - old_dimensionality  # get the dimensionality of this q
            old_dimensionality = s  # remember the old dimensionality

        ## stack matricies as necessary
        A = np.array([]).reshape((0, N))
        for q, keep in d.items():
            A = np.vstack((A, self.Pt_complete(int(q), N, keep,
                                               self._basis_type)))
            # A = np.vstack((A, self.Cq_complete(int(q), N)))
        return (A, d)

    @staticmethod
    def Cq(q, s=0, repetitions=1, type='real'):
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

    def Cq_complete(self, q, N=None):
        if N is None:
            N = q  # make it one period by default
        # et = eulers_totient(q)
        et = q
        matrix = np.zeros((et, int(N)))
        cq = self.Cq(q)
        for i in range(et):
            cq_rotated = np.roll(cq, i)
            repetitions = int(np.ceil(N / q))
            matrix[i] = np.tile(cq_rotated, repetitions)[:N]
        return matrix

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
                matrix[i] = QOPeriod.Pp(p, i, repetitions)[:int(N)]
            elif type == 'ramanujan':
                matrix[i] = QOPeriod.Cq(p, i, repetitions, 'real')[:int(N)]
        if keep:
            matrix = matrix[:keep]
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
