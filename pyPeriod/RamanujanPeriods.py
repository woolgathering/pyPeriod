import numpy as np
from functools import reduce
import itertools
from pyPeriod import QOPeriods
from typing import Callable
from warnings import warn
from .utils import get_factors, get_primes, phi, rms

# from numba import jit, vectorize, float64, float32, int64, int32 # not included yet

class RamanujanPeriods:
    def __init__(self, basis_type="natural"):
        self._basis_type = basis_type
        self._output = None
        self._verbose = None

    def find_periods(self, x:np.ndarray, min_length:int=2, max_length:int=None, select_periods:Callable=None) -> np.ndarray:
        if not max_length:
            max_length = len(x) // 3

        norms = np.zeros(max_length + 1)
        for p in range(min_length, max_length + 1):
            if self._verbose:
                print("Processing for period {}".format(p))
            basis = self.Cq_complete(p, len(x))
            projection = self.project(x, basis)
            output = np.sum(projection, 0)
            norms[p] = np.sum(np.power(output, 2))
            if self._verbose:
                print("\tThis norm: {}".format(norms[p]))

        if select_periods:
            if hasattr(select_periods, "__call__"):
                return select_periods(norms)
        else:
            return norms

    def find_periods_with_weights(
        self, x, min_length=2, max_length=None, thresh=0.2, **kwargs
    ):
        norms = self.find_periods(x, min_length, max_length, select_periods=kwargs.get('select_periods', None))

        # if test_function is set, thresh is overridden
        if "test_function" in kwargs.keys():
            test_function = kwargs["test_function"]
        else:
            test_function = lambda x: np.argwhere(
                x / np.abs(np.max(x)) > thresh
            ).flatten()

        periods = test_function(norms)

        if self._verbose:
            print("Found periods {}".format(periods))

        basis_matricies, basis_dictionary = self.get_subspaces(
            periods, len(x)
        )  # get the subspaces and a dictionary describing its construction
        resconst, output_weights = self.solve_quadratic(
            x, basis_matricies
        )  # get the new residual and do it again
        res = x - resconst  # get the residual

        output_bases = {
            "periods": periods,
            "norms": norms[periods],
            "subspaces": basis_matricies,
            "weights": output_weights,
            "basis_dictionary": basis_dictionary,
        }
        self._output = output_bases
        return (output_bases, res)

    def project(self, x, basis):
        proj_complete = np.zeros(basis.shape, dtype=np.float32)
        for i, row in enumerate(basis):
            row = row / np.max(row)
            proj_complete[i] = np.dot(x, row) * row
        return proj_complete

    def Cq(self, q, s=0, repetitions=1, type="real"):
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
            warn("Return type invalid, defaulting to 'real'")
            return np.real(vec)

    def Cq_complete(self, q, N=None, normalize=True):
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
            if normalize:
                matrix[i] /= np.linalg.norm(matrix[i])
        return matrix
