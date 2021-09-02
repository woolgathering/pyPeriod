"""
Utilities for working with periodicities
"""

import numpy as np
from scipy.interpolate import interp1d
import itertools

def resample(sig, new_size=1024, kind='quadratic', dtype=np.float64):
    interp = interp1d(np.linspace(0, len(sig) - 1, len(sig), dtype=np.int16),
                      sig,
                      kind=kind)
    x = np.linspace(0, len(sig) - 1, new_size, dtype=np.float64)
    output = np.zeros(new_size, dtype=dtype)
    for i, value in enumerate(x):
        output[i] = interp(value)
    return output


def project_float(data, p=2, wavetable_len=1024, dtype=np.float64):
    def wrap(a, length):
        return a % length

    def interp(idx, signal):
        trunc = int(idx)
        frac = idx - trunc

        x0 = wrap(trunc - 1, signal.size - 1)
        x1 = wrap(trunc, signal.size - 1)
        x2 = wrap(trunc + 1, signal.size - 1)
        x3 = wrap(trunc + 2, signal.size - 1)

        # calculate the coefficients
        a = -0.5 * signal[x0] + 1.5 * signal[x1] - 1.5 * signal[
            x2] + 0.5 * signal[x3]
        b = signal[x0] - 2.5 * signal[x1] + 2 * signal[x2] - 0.5 * signal[x3]
        c = -0.5 * signal[x0] + 0.5 * signal[x2]
        d = signal[x1]

        return a * (frac**3) + b * (frac**2) + c * frac + d

    if p == 1:
        return data
    else:
        # interp = interp1d(np.linspace(0, data.size-1, data.size, dtype=np.float32), data, kind='quadratic', fill_value='extrapolate')
        # wavetable_len = 1024 # length of the wavetable. Basically, wavetable_len/p is the upsample factor.
        N = len(data)
        projection = np.zeros(N, dtype)
        single_period = np.zeros(
            wavetable_len, dtype
        )  # <---- first distortion occurs here. Real period is wavetable_len, logical period is p
        fac = p / wavetable_len  # get the factor for conversion between wavetable_len and p
        for s in range(wavetable_len):
            this_sum = 0
            count = 0
            start = s * fac
            for n in range(math.ceil(N / p)):
                idx = start + (n * p)
                if idx < data.size - 1:
                    this_sum += interp(idx, data)
                    count += 1
            single_period[s] = this_sum / count  # remember it

        # now we oscillate through the above wavetable at period p
        inc = wavetable_len / p
        # interp = interp1d(np.linspace(0, single_period.size-1, single_period.size, dtype=np.float32), single_period, kind='quadratic',fill_value='extrapolate')
        for i in range(projection.size):
            idx = np.mod(i * inc, wavetable_len)
            projection[i] = interp(idx, single_period)

    return projection  # return the project

################################################################################
"""
Functions for the quadratic optimization method.
"""
################################################################################


def solve_quadratic(x, bases, gcds=None):
    if gcds is None:
        A = bases
    else:
        A = np.vstack(tuple([*bases, *gcds]))  # stack it
    A_prime = np.matmul(A, A.T)  # multiply by its transpose
    W = np.matmul(A, x)  # multiply by the data
    output = np.linalg.solve(A_prime, W)  # actually solve it
    # output, istop, itn, normr = lsqr(A_prime, W)[:4] # actually solve it
    reconstructed = np.matmul(A.T, output)  # reconstruct the output
    return (reconstructed, output)

def get_gcds_old(input1, input2=None):
    if isinstance(input1, int):
        gcds = []
        for value in input2:
            gcds.append(np.gcd(input1, value))
        gcds = np.array(gcds)
    else:
        gcds = np.array([np.gcd(l[0], l[1]) for l in input1])

    gcds = np.unique(np.array([gcds]))  # remove duplicates
    gcds = np.sort(gcds)  # sort ascending
    return gcds

def get_gcds(pers, gcds, concat=True):
    new_p = pers[-1]
    l = np.concatenate((pers[:-1], gcds))  # gcdsmake a new list
    new_gcds = np.zeros(len(l), dtype=np.int32)
    for i, p in enumerate(l):
        new_gcds[i] = np.gcd(new_p, p, dtype=np.int32)
    if concat:
        return np.unique(np.concatenate((new_gcds, gcds)))
    else:
        return np.unique(new_gcds)

def Pt_complete(p, N=1, lop_off=None):
    repetitions = int(np.ceil(N / p))
    matrix = np.zeros((p, int(N)))
    for i in range(p):
        matrix[i] = Periods.Pp(p, i, repetitions)[:int(N)]

    if lop_off:
        matrix = matrix[:-lop_off]
    return matrix

def Pp(p, s, repetitions=1):
    vec = np.zeros(p)
    for i in range(p):
        if (i - s) % p == 0:
            vec[i] = 1
    return np.tile(vec, repetitions)

def reduce_rows(A):
    AA = A[0]
    rank = np.linalg.matrix_rank(AA) # should be 1
    for row in A[1:]:
        aa = np.vstack((AA, row))
        if np.linalg.matrix_rank(aa) > int(row):
            AA = aa
            row = np.linalg.matrix_rank(aa)
    return AA

def stack_pairwise_gcd_subspaces(periods):
    subspace = []
    all_pairs = list(itertools.combinations(periods, 2))
    for pair in all_pairs:
        gcd = np.gcd(pair[0], pair[1])
        row = np.array([], dtype=np.int64)
        for p in periods:
            if p in set(pair):
                if p == pair[0]:
                    ss = Pp(gcd, 0, int(p/gcd)) * -1
                else:
                    ss = Pp(gcd, 0, int(p/gcd))
            else:
                ss = np.zeros(p)

            row = np.append(row, ss)

        subspace.append(row)
        for i in range(1, gcd):
            subspace.append(np.roll(row, i))

    return np.vstack(tuple(subspace))

def get_original_periods(periods, concatenated_periods):
    A = stack_pairwise_gcd_subspaces(periods)
    A = reduce_rows(A)
    reconst, coeffs = solve_quadratic(concatenated_periods, A)
    actual_concatenated_periods = concatenated_periods - reconst
    actual_periods = []
    for i,p in enumerate(periods):
        start = np.sum(periods[:i])
        actual_periods.append(actual_concatenated_periods[start:start+p])
    return tuple(actual_periods)
