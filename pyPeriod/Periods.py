"""
Periodicity transform as described by Sethares and Staley in \"Periodicity  Transforms\",
IEEE  Transactions  on Signal  Processing, Vol. 47, No. 11, November 1999
(https://sethares.engr.wisc.edu/paperspdf/pertrans.pdf)
"""

import numpy as np
from warnings import warn
from .utils import get_primes, get_factors

################################################################################
################################################################################
################################################################################
class Periods:
    """
    A class implementing the periodicity transform as described by Sethares and Staley in \"Periodicity  Transforms\", IEEE  Transactions  on Signal  Processing, Vol. 47, No. 11, November 1999 (https://sethares.engr.wisc.edu/paperspdf/pertrans.pdf).

    ...

    Attributes
    ----------
    PRIMES : set
        a class variable to hold a set of primes

    Methods
    -------
    project(data, p=2, trunc_to_integer_multiple=False, orthogonalize=False, return_single_period=False)
        Projects the data onto a lower-dimensional space.
    periodic_norm(x, p=None)
        Calculates the periodic norm of the input vector.
    small_to_large(data, thresh=0.1, n_periods=None)
        Finds periods in the data from small to large.
    best_correlation(data, num=5, max_length=None, ratio=0.01)
        Finds the best correlation in the data.
    best_frequency(data, win_size=None, num=5)
        Finds the best frequency in the data.
    m_best(data, num=5, max_length=None, min_length=2)
        Finds the M-best periods in the data.
    m_best_gamma(data, num=5, max_length=None, min_length=2)
        Finds the M-best gamma periods in the data.
    _m_best_meta(data, type, num=5, max_length=None, min_length=2)
        Helper function for m_best and m_best_gamma.
    """

    PRIMES = set(get_primes(10000))  # a class variable to hold a set of primes

    def __init__(
        self, trunc_to_integer_multiple: bool = False, orthogonalize: bool = False
    ):
        """
        Constructs all the necessary attributes for the Periods object.

        Parameters
        ----------
            trunc_to_integer_multiple : bool, optional
                Whether or not to truncate the window to fit an integer multiple of the period (default is False)
            orthogonalize : bool, optional
                Whether or not to orthogonalize the projections (default is False)
        """
        self._trunc_to_integer_multiple = trunc_to_integer_multiple
        self._orthogonalize = orthogonalize
        # self._window = window
        # if window:
        #     data = data * np.hanning(len(data))

    def project(
        self,
        data: np.ndarray,
        p: int = 2,
        return_single_period: bool = False,
    ) -> np.ndarray:
        """
        Projects the data onto a lower-dimensional space.

        Parameters
        ----------
            data : np.ndarray
                The data to be projected.
            p : int, optional
                The period (default is 2).
            return_single_period : bool, optional
                Whether or not to return a single period (default is False).

        Returns
        -------
            np.ndarray
                The projected data.
        """
        projection = self._project(data, p)

        # a faster, cleaner way to orthogonalize that is equivalent to the method
        # presented in "Orthogonal, exactly periodic subspace decomposition" (D.D.
        # Muresan, T.W. Parks), 2003. Setting trunc_to_integer_multiple gives a result
        # that is almost exactly identical (within a rounding error; i.e. 1e-6).
        # For the outputs of each to be identical, the input MUST be the same length
        # with DC removed since the algorithm in Muresan truncates internally and
        # here we allow the output to assume the dimensions of the input. See above
        # line of code.
        if self._orthogonalize:
            for f in get_factors(p, remove_1_and_n=True):
                if f in self.PRIMES:
                    # remove the projection at p/prime_factor, taking care not to remove things twice.
                    projection = projection - self._project(
                        projection, int(p / f), False
                    )

        if return_single_period:
            return projection[0:p]  # just a single period
        else:
            return projection  # the whole thing

    def _project(self, data: np.ndarray, p: int) -> np.ndarray:
        cp = data.copy()
        samples_short = int(
            np.ceil(len(cp) / p) * p - len(cp)
        )  # calc how many samples short for rectangle
        cp = np.pad(cp, (0, samples_short))  # pad it
        cp = cp.reshape(int(len(cp) / p), p)  # reshape it to a rectangle

        if self._trunc_to_integer_multiple:
            if samples_short == 0:
                single_period = np.mean(cp, 0)  # don't need to omit the last row
            else:
                single_period = np.mean(
                    cp[:-1], 0
                )  # just take the mean of the truncated version and output a single period
        else:
            ## this is equivalent to the method presented in Sethares but significantly faster
            ## do the mean manually. get the divisors from the last row since the last samples_short values will be one less than the others
            divs = np.zeros(cp.shape[1])
            for i in range(cp.shape[1]):
                if i < (cp.shape[1] - samples_short):
                    divs[i] = cp.shape[0]
                else:
                    divs[i] = cp.shape[0] - 1
            single_period = np.sum(cp, 0) / divs  # get the mean manually

        return np.tile(single_period, int(data.size / p) + 1)[
            : len(data)
        ]  # extend the period and take the good part

    def periodic_norm(self, x: np.ndarray, p: int = None) -> float:
        """
        Calculates the periodic norm of the input vector.

        Parameters
        ----------
            x : np.ndarray
                The input vector.
            p : int, optional
                The period (default is None).

        Returns
        -------
            float
                The periodic norm of the input vector.
        """
        if p:
            return (np.linalg.norm(x) / np.sqrt(len(x))) / np.sqrt(p)
        else:
            return np.linalg.norm(x) / np.sqrt(len(x))

    ##############################################################################
    ### Actual period-finding algorithms
    ##############################################################################
    def small_to_large(
        self, data: list, thresh: float = 0.1, n_periods: int = None
    ) -> tuple[list, list, list]:
        """
        Implementation of the small-to-large algorithm.

        Parameters
        ----------
            data : list
                The data to findperiods in.
            thresh : float, optional
                The threshold (default is 0.1).
            n_periods : int, optional
                The number of periods (default is None).

        Returns
        -------
            tuple[list, list, list]
                The periods, powers, and bases. All items in the lists correspond by index. The periods and powers list are both the length of the number of periods found in the data. The bases is a list of lists where the first dimension is the number of periods found and each base at index `i` is a list of period `period[i]` with power `power[i]` and is the same length as the input data.
        """
        periods = []
        powers = []
        bases = []
        data_norm = self.periodic_norm(data)
        residual = data.copy()
        if n_periods is None:
            n_periods =len(data) // 2
        for p in range(2, n_periods + 1):
            base = self.project(
                residual, p, self._trunc_to_integer_multiple, self._orthogonalize
            )  # project
            this_residual = residual - base  # get the residual
            imposed_norm = (
                self.periodic_norm(residual) - self.periodic_norm(this_residual)
            ) / data_norm
            if imposed_norm > thresh:
                # save it
                residual = this_residual
                periods.append(p)
                powers.append(imposed_norm)
                bases.append(base)
        return (periods, powers, bases)

    def best_correlation(
        self, data: list, num: int = 5, max_length: int = None, ratio: float = 0.01
    ) -> tuple[list, list, list]:
        """
        Implementation of the best correlation algorithm.

        Parameters
        ----------
            data : list
                The data to find the best correlation in.
            num : int, optional
                The number of correlations (default is 5).
            max_length : int, optional
                The maximum length (default is None).
            ratio : float, optional
                The ratio (default is 0.01).

        Returns
        -------
            tuple[list, list, list]
                The periods, norms, and bases.
        """
        if max_length is None:
            max_length = len(data) // 3
        periods = np.zeros(num, dtype=np.uint32)
        norms = np.zeros(num)
        bases = np.zeros((num, len(data)))
        og_norm = self.periodic_norm(data)  # original gangsta norm
        old_norm = og_norm
        data_copy = data.copy()

        for i in range(num):
            # check correlation
            max_cor = 0
            max_period = None
            for p in range(2, max_length):
                # p is the period
                cor = 0
                for s in range(0, p):
                    cor = abs(sum(data_copy[s::p]))
                    if cor > max_cor:
                        max_cor = cor
                        max_period = p

            # check to see if it's actually any good
            base = self.project(
                data_copy,
                max_period,
                self._trunc_to_integer_multiple,
                self._orthogonalize,
            )
            data_copy = data_copy - base
            this_norm = self.periodic_norm(data_copy)
            norm_test = (old_norm - this_norm) / og_norm
            if norm_test > ratio:
                periods[i] = max_period
                norms[i] = norm_test
                bases[i] = base
                old_norm = this_norm

        return (periods, norms, bases)

    def best_frequency(
        self, data: list, win_size: int = None, num: int = 5
    ) -> tuple[list, list, list]:
        """
        Implementation of the best frequency algorithm.

        Parameters
        ----------
            data : list
                The data to find the best frequency in.
            win_size : int, optional
                The window size (default is None: i.e. the length of the).
            num : int, optional
                The number of frequencies (default is 5).

        Returns
        -------
            tuple[list, list, list]
                The periods, powers, and bases. All items in the lists correspond by index. The periods and powers list are both the length of the number of periods found in the data. The bases is a list of lists where the first dimension is the number of periods found and each base at index `i` is a list of period `period[i]` with power `power[i]` and is the same length as the input data.
        """
        if win_size is None:
            win_size = len(data)
        elif win_size < len(data):
            warn(
                "win_size is smaller than the input signal length. It will be truncated and information will be lost."
            )

        periods = np.zeros(num, dtype=np.uint32)
        norms = np.zeros(num)
        bases = np.zeros((num, len(data)))
        data_copy = data.copy()

        for i in range(num):
            mags = np.abs(
                np.fft.rfft(data_copy, win_size)
            )  # we only need the magnitude of the positive freqs
            p = (2 * win_size) / np.argmax(mags)  # get the period
            p = int(np.round(p))  # round it and make it an integer
            base = self.project(
                data_copy, p, self._trunc_to_integer_multiple, self._orthogonalize
            )  # project
            periods[i] = p  # remember it
            norms[i] = self.periodic_norm(base)
            bases[i] = base
            data_copy = data_copy - base  # remove it

        powers = norms / self.periodic_norm(data)
        return (periods, powers, bases)

    """
    M-best family. Note that for this, orthogonalize is set to FALSE regardless of
    what the instance holds since it makes no sense to do step two if things are
    orthogonalized.

    A warning is thrown.
    """

    def m_best(
        self, data: list, num: int = 5, max_length: int = None, min_length: int = 2
    ) -> tuple[list, list, list]:
        """
        Implementation of the M-best algorithm.

        Parameters
        ----------
            data : list
                The data to find the M-best periods in.
            num : int, optional
                The number of periods (default is 5).
            max_length : int, optional
                The maximum length (default is None).
            min_length : int, optional
                The minimum length (default is 2).

        Returns
        -------
            tuple[list, list, list]
                The periods, powers, and bases. All items in the lists correspond by index. The periods and powers list are both the length of the number of periods found in the data. The bases is a list of lists where the first dimension is the number of periods found and each base at index `i` is a list of period `period[i]` with power `power[i]` and is the same length as the input data.
        """
        return self._m_best_meta(data, None, num, max_length, min_length)

    def m_best_gamma(
        self, data: list, num: int = 5, max_length: int = None, min_length: int = 2
    ) -> tuple[list, list, list]:
        """
        Implementation of the M-best gamma algorithm.

        Parameters
        ----------
            data : list
                The data to find the M-best gamma periods in.
            num : int, optional
                The number of periods (default is 5).
            max_length : int, optional
                The maximum length (default is None).
            min_length : int, optional
                The minimum length (default is 2).

        Returns
        -------
            tuple[list, list, list]
                The periods, powers, and bases. All items in the lists correspond by index. The periods and powers list are both the length of the number of periods found in the data. The bases is a list of lists where the first dimension is the number of periods found and each base at index `i` is a list of period `period[i]` with power `power[i]` and is the same length as the input data.
        """
        return self._m_best_meta(data, "gamma", num, max_length, min_length)

    def _m_best_meta(
        self, data: list, type, num=5, max_length=None, min_length=2
    ) -> tuple[list, list, list]:
        """
        Helper function for m_best and m_best_gamma. Should not be called directly by the user.

        Parameters
        ----------
            data : list
                The data to find the M-best periods in.
            type : str
                The type of M-best to find.
            num : int, optional
                The number of periods (default is 5).
            max_length : int, optional
                The maximum length (default is None).
            min_length : int, optional
                The minimum length (default is 2).

        Returns
        -------
            tuple[list, list, list]
                The periods, powers, and bases. All items in the lists correspond by index. The periods and powers list are both the length of the number of periods found in the data. The bases is a list of lists where the first dimension is the number of periods found and each base at index `i` is a list of period `period[i]` with power `power[i]` and is the same length as the input data.
        """

        # remind the user that orthogonalize has no effect here
        if self.orthogonalize:
            warn("`Orthogonalize = True` has no effect in M-best.")

        if max_length is None:
            max_length = len(data) // 3
        data_copy = data.copy()
        periods = np.zeros(num, dtype=np.uint32)
        norms = np.zeros(num)
        bases = np.zeros((num, len(data)))
        skip_periods = []  # skip periods that continue to show up and slow things down

        # step 1
        i = 0
        iters = 0
        while i < num:
            max_norm = 0
            max_period = 0
            max_base = None
            # print ('Number {}'.format(i))
            for p in range(min_length, max_length + 1):
                # base = self.project(data_copy, p,
                #                     self._trunc_to_integer_multiple, False)
                base = self.project(
                    data_copy, p, self._trunc_to_integer_multiple, self._orthogonalize
                )
                if type is None:
                    p_norm = self.periodic_norm(base)  # m-best
                else:
                    p_norm = self.periodic_norm(base, p)  # m-best gamma

                if (p_norm > max_norm) and (p not in set(skip_periods)):
                    max_period = p
                    max_norm = p_norm
                    max_base = base

            # if (max_period in set(periods)) and (self.periodic_norm(data_copy)>0.01):
            if (max_period in set(periods)) and (iters < 10):
                idx = np.where(periods == max_period)[0]
                bases[idx] += max_base
                norms[
                    idx
                ] += max_norm  # probably need to recalculate the norm but leave it for now
                iters += 1
            elif (max_period in set(periods)) and (iters >= 10):
                skip_periods.append(
                    max_period
                )  # remember to skip this one in the future
                iters = 0
            else:
                periods[i] = max_period
                norms[i] = max_norm
                bases[i] = max_base
                i += 1  # only increment i if we add a new period
                iters = 0

            data_copy = data_copy - max_base  # remove the best one and do it again

        # step 2
        changed = True
        while changed:
            i = 0
            while i < num:
                changed = False
                max_norm = 0
                max_period = None
                max_base = None
                facs = get_factors(periods[i], remove_1_and_n=True)
                for f in facs:
                    base = self.project(
                        bases[i],
                        f,
                        self._trunc_to_integer_multiple,
                        self._orthogonalize,
                    )
                    if type is None:
                        norm = self.periodic_norm(base)
                    else:
                        norm = self.periodic_norm(base, p)
                    if norm > max_norm:
                        max_period = f
                        max_norm = norm
                        max_base = base

                if max_period not in periods and max_period is not None:
                    xQ = max_base
                    xq = bases[i] - xQ
                    nQ = max_norm
                    if type is None:
                        nq = self.periodic_norm(base)
                    else:
                        nq = self.periodic_norm(base, p)
                    min_q = min(norms)
                    if (
                        (nq + nQ) > (norms[num - 1] + norms[i])
                        and (nq > min_q)
                        and (nQ > min_q)
                    ):
                        changed = True

                        # keep the old one but now it's weakened. Replace values.
                        bases[i] = xq
                        norms[i] = nq
                        # periods[i] = # period is the same as before, just has a strong factor removed

                        # now pop in our new one at one higher in the list (i). This grows the list by 1
                        bases = np.insert(bases, i, max_base, 0)
                        norms = np.insert(norms, i, nQ)
                        periods = np.insert(periods, i, max_period)

                        # remove the last (weakest) basis vector
                        bases = bases[:num]
                        norms = norms[:num]
                        periods = periods[:num]
                    else:
                        i += 1
                else:
                    i += 1

        powers = norms / self.periodic_norm(data)
        return (periods, powers, bases)

    ######################
    # Properties ########
    ######################
    def trunc_to_integer_multiple():
        doc = "Boolean on whether or not to truncate the window to fit an integer \
      multiple of the period"

        def fget(self):
            return self._trunc_to_integer_multiple, self._orthogonalize

        def fset(self, value):
            self._trunc_to_integer_multiple, self._orthogonalize = value

        return locals()

    trunc_to_integer_multiple = property(**trunc_to_integer_multiple())

    def orthogonalize():
        doc = "Boolean on whether or not to orthogonalize the projections."

        def fget(self):
            return self._orthogonalize

        def fset(self, value):
            self._orthogonalize = value

        return locals()

    orthogonalize = property(**orthogonalize())

    def window():
        doc = "The window property."

        def fget(self):
            return self._window

        def fset(self, value):
            self._window = value

        return locals()

    window = property(**window())
