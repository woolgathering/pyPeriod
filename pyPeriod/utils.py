import numpy as np
from functools import reduce

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

def rms(x: list) -> float:
    """
    Computes the Root Mean Square (RMS) of a list of numbers.

    Parameters
    ----------
    x : list
        A list of numerical values.

    Returns
    -------
    float
        The RMS value of the input list.
    """
    return np.sqrt(np.sum(np.power(x, 2)) / len(x))


def flatten(t: list) -> list:
    """
    Flattens a nested list.

    Parameters
    ----------
    t : list
        A nested list.

    Returns
    -------
    list
        A flattened list containing all elements from the input nested list.
    """
    return [item for sublist in t for item in sublist]


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
            ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0),
        )
    )
    if remove_1:
        facs.remove(1)
    if remove_n and n != 1:
        facs.remove(n)
    return facs


def reduce_rows(A: np.ndarray) -> np.ndarray:
    """
    Reduces the rows of a 2D array to its linearly independent subset.

    Parameters
    ----------
    A : np.ndarray
        A 2D numpy array.

    Returns
    -------
    np.ndarray
        A 2D numpy array containing only the linearly independent rows of the input array.
    """
    AA = A[0]
    rank = np.linalg.matrix_rank(AA)
    for row in A[1:]:
        aa = np.vstack((AA, row))
        if np.linalg.matrix_rank(aa) > rank:
            AA = aa
            rank = np.linalg.matrix_rank(aa)
    return AA


def get_primes(max: int = 1000000) -> list[int]:
    """
    Generate all prime numbers up to a given maximum.

    Parameters
    ----------
    max : int, optional
        The maximum number up to which to generate primes, by default 1000000.

    Returns
    -------
    list[int]
        A list of all prime numbers up to the given maximum.
    """
    primes = np.arange(3, max + 1, 2)
    isprime = np.ones((max - 1) // 2, dtype=bool)
    for factor in primes[: int(np.sqrt(max)) // 2]:
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