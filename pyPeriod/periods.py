"""
Periodicity transform as described by Sethares and Staley in \"Periodicity  Transforms\",
IEEE  Transactions  on Signal  Processing, Vol. 47, No. 11, November 1999
(https://sethares.engr.wisc.edu/paperspdf/pertrans.pdf)
"""

import numpy as np
from functools import reduce
import math
from warnings import warn
import itertools

################################################################################
################################################################################
################################################################################
def get_factors(n, remove_1_and_n=False):
  facs = set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
  # get rid of 1 and the number itself
  if remove_1_and_n:
    facs.remove(1)
    facs.remove(n)
  return facs

def get_primes(max=1000000):
  primes=np.arange(3,max+1,2)
  isprime=np.ones(int((max-1)/2),dtype=bool)
  for factor in primes[:int(np.sqrt(max))]:
    if isprime[int((factor-2)/2)]: isprime[int((factor*3-2)/2)::factor]=0
  return np.sort(np.insert(primes[isprime],0,2))

def rms(x):
  return np.sqrt(np.sum(np.power(x, 2))/len(x))
################################################################################
################################################################################
################################################################################

class Periods:

  PRIMES = set(get_primes(10000)) # a class variable to hold a set of primes

  def __init__(self, data, trunc_to_integer_multiple=False, orthogonalize=False, window=False):
    super(Periods, self).__init__()
    self._data = np.array(data)
    self._trunc_to_integer_multiple = trunc_to_integer_multiple
    self._orthogonalize = orthogonalize
    self._window = window
    if window:
      self._data = self._data * np.hanning(len(data))

  @staticmethod
  def project(data, p=2, trunc_to_integer_multiple=False, orthogonalize=False, return_single_period=False):
    cp = data.copy()
    samples_short = int(np.ceil(len(cp)/p)*p - len(cp)) # calc how many samples short for rectangle
    cp = np.pad(cp, (0,samples_short)) # pad it
    cp = cp.reshape(int(len(cp)/p), p) # reshape it to a rectangle

    if trunc_to_integer_multiple:
      if samples_short==0:
        single_period = np.mean(cp, 0) # don't need to omit the last row
      else:
        single_period = np.mean(cp[:-1], 0) # just take the mean of the truncated version and output a single period
    else:
      ## this is equivalent to the method presented in the paper but significantly faster ##
      # do the mean manually. get the divisors from the last row since the last samples_short values will be one less than the others
      divs = np.zeros(cp.shape[1])
      for i in range(cp.shape[1]):
        if i < (cp.shape[1]-samples_short):
          divs[i] = cp.shape[0]
        else:
          divs[i] = cp.shape[0]-1
      single_period = np.sum(cp, 0) / divs # get the mean manually

    projection = np.tile(single_period, int(data.size/p)+1)[:len(data)] # extend the period and take the good part

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
        if f in Periods.PRIMES:
          # remove the projection at p/prime_factor, taking care not to remove things twice.
          projection = projection - Periods.project(projection, int(p/f), trunc_to_integer_multiple, False)

    if return_single_period:
      return projection[0:p] # just a single period
    else:
      return projection # the whole thing

  @staticmethod
  def periodic_norm(x, p=None):
    if p:
      return (np.linalg.norm(x) / np.sqrt(len(x))) / np.sqrt(p)
    else:
      return np.linalg.norm(x) / np.sqrt(len(x))

  ##############################################################################
  ### Actual period-finding algorithms
  ##############################################################################
  def small_to_large(self, thresh=0.1, n_periods=None):
    periods = []
    powers = []
    bases = []
    data_norm = self.periodic_norm(self.data)
    residual = self.data.copy()
    if n_periods is None:
      n_periods = math.floor(len(self.data)/2)
    for p in range(2, n_periods+1):
      base = self.project(residual, p, self._trunc_to_integer_multiple, self._orthogonalize) # project
      this_residual = residual - base # get the residual
      imposed_norm = (self.periodic_norm(residual) - self.periodic_norm(this_residual)) / data_norm
      if imposed_norm > thresh:
        # save it
        residual = this_residual
        periods.append(p)
        powers.append(imposed_norm)
        bases.append(base)
    return (periods, powers, bases)

  def best_correlation(self, num=5, max_length=None, ratio=0.01):
    if max_length is None:
      max_length = math.floor(len(self._data)/3)
    periods = np.zeros(num, dtype=np.uint32)
    norms = np.zeros(num)
    bases = np.zeros((num, len(self._data)))
    og_norm = self.periodic_norm(self._data) # original gangsta norm
    old_norm = og_norm
    data_copy = self._data.copy()

    for i in range(num):
      # check correlation
      max_cor = 0
      max_period = None
      for p in range(2,max_length):
        # p is the period
        cor = 0
        for s in range(0, p):
          cor = abs(sum(data_copy[s::p]))
          if cor > max_cor:
            max_cor = cor
            max_period = p

      # check to see if it's actually any good
      base = self.project(data_copy, max_period, self._trunc_to_integer_multiple, self._orthogonalize)
      data_copy = data_copy - base
      this_norm = self.periodic_norm(data_copy)
      norm_test = ((old_norm - this_norm) / og_norm)
      if norm_test > ratio:
        periods[i] = max_period
        norms[i] = norm_test
        bases[i] = base
        old_norm = this_norm

    return (periods, norms, bases)

  def best_frequency(self, win_size=None, num=5):
    if win_size is None:
      win_size = len(self._data)
    elif win_size < len(self._data):
      warn('win_size is smaller than the input signal length. It will be truncated and information will be lost.')

    periods = np.zeros(num, dtype=np.uint32)
    norms = np.zeros(num)
    bases = np.zeros((num, len(self._data)))
    data_copy = self._data.copy()

    for i in range(num):
      mags = np.abs(np.fft.rfft(data_copy, win_size)) # we only need the magnitude of the positive freqs
      p = (2*win_size)/np.argmax(mags) # get the period
      p = int(np.round(p)) # round it and make it an integer
      base = self.project(data_copy, p, self._trunc_to_integer_multiple, self._orthogonalize) # project
      periods[i] = p # remember it
      norms[i] = self.periodic_norm(base)
      bases[i] = base
      data_copy = data_copy - base # remove it

    powers = norms / self.periodic_norm(self._data)
    return (periods, powers, bases)

  """
  M-best family. Note that for this, orthogonalize is set to FALSE regardless of
  what the instance holds since it makes no sense to do step two if things are
  orthogonalized.

  A warning is thrown.
  """
  def m_best(self, num=5, max_length=None, min_length=2):
    return self._m_best_meta(None, num, max_length, min_length)

  def m_best_gamma(self, num=5, max_length=None, min_length=2):
    return self._m_best_meta('gamma', num, max_length, min_length)

  def _m_best_meta(self, type, num=5, max_length=None, min_length=2):
    # remind the user that orthogonalize has no effect here
    if self.orthogonalize:
      warn("`Orthogonalize = True` has no effect in M-best.")

    if max_length is None:
      max_length = math.floor(len(self.data)/3)
    data_copy = self.data.copy()
    periods = np.zeros(num, dtype=np.uint32)
    norms = np.zeros(num)
    bases = np.zeros((num, len(self.data)))
    skip_periods = [] # skip periods that continue to show up and slow things down

    # step 1
    i = 0
    iters = 0
    while i < num:
      max_norm = 0
      max_period = 0
      max_base = None
      # print ('Number {}'.format(i))
      for p in range(min_length,max_length+1):
        base = self.project(data_copy, p, self._trunc_to_integer_multiple, False)
        if type is None:
          p_norm = self.periodic_norm(base) # m-best
        else:
          p_norm = self.periodic_norm(base, p) # m-best gamma

        if (p_norm > max_norm) and (p not in set(skip_periods)):
          max_period = p
          max_norm = p_norm
          max_base = base

      # if (max_period in set(periods)) and (self.periodic_norm(data_copy)>0.01):
      if (max_period in set(periods)) and (iters<10):
        idx = np.where(periods==max_period)[0]
        bases[idx] += max_base
        norms[idx] += max_norm # probably need to recalculate the norm but leave it for now
        iters += 1
      elif(max_period in set(periods)) and (iters>=10):
        skip_periods.append(max_period) # remember to skip this one in the future
        iters = 0
      else:
        periods[i] = max_period
        norms[i] = max_norm
        bases[i] = max_base
        i += 1 # only increment i if we add a new period
        iters = 0

      data_copy = data_copy - max_base # remove the best one and do it again

    # step 2
    changed = True
    while changed:
      i = 0
      while i<num:
        changed = False
        max_norm = 0
        max_period = None
        max_base = None
        facs = get_factors(periods[i], remove_1_and_n=True)
        for f in facs:
          base = self.project(bases[i], f, self._trunc_to_integer_multiple, False)
          if type is None:
            norm = self.periodic_norm(base)
          else:
            norm = self.periodic_norm(base, p)
          # norm = func_name(base, p)
          if norm>max_norm:
            max_period = f
            max_norm = norm
            max_base = base

        if max_period not in periods and max_period is not None:
          # xQ = self.project(bases[i], max_period) # redundant
          xQ = max_base
          xq = bases[i] - xQ
          # nQ = func_name(xQ) # redundant
          nQ = max_norm
          # nq = func_name(xq, p)
          if type is None:
            nq = self.periodic_norm(base)
          else:
            nq = self.periodic_norm(base, p)
          min_q = min(norms)
          if (nq+nQ) > (norms[num-1]+norms[i]) and (nq>min_q) and (nQ>min_q):
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

    powers = norms/self.periodic_norm(self.data)
    return (periods, powers, bases)

  ##############################################################################
  ##############################################################################
  ##############################################################################
  """
  Using quadradic optimization to find the best periods.

  The biggest trick is finding a good min_length (smallest acceptable period that
  is not a GCD) that describes the data well and does not introduce small periods
  (high frequencies) in the found periods. The most promising method so far (for
  sound, at least) is to take the spectral centroid of the data, weighting low
  frequencies by dividing the magnitudes by the bin number and use the result
  as the smallest allowable period.
  """
  def quadratic_optimization(self, num=5, thresh=0.05, min_length=2, max_length=None):
    N = len(self._data)
    if max_length is None:
      max_length = int(math.floor(N/3))
    periods = np.zeros(num, dtype=np.uint32)
    norms = np.zeros(num)
    bases = np.zeros((num, N))
    res = self._data # copy
    rms_thresh = rms(self._data) * thresh

    # these get recomputed (!!!) each time but stick them here so we can exit whenever
    basis_matricies = None # a list to later be changed into a tuple
    nonzero_periods = None # periods that are not 0
    gcd_matricies = None # space to put the subspaces
    gcd_added = None # remember what we've added (same index as gcd_matricies)

    for i in range(num):
      if rms(res) > rms_thresh:
        best_p = 0
        best_norm = 0
        best_base = None
        for p in range(min_length, max_length+1):
          this_base = self.project(res, p, self._trunc_to_integer_multiple, False)
          # this_norm = self.periodic_norm(this_base)
          this_norm = self.periodic_norm(this_base,p)
          if this_norm > best_norm:
            best_p = p
            best_norm = this_norm
            best_base = this_base

        # now that we've found the strongest period in this run, set them
        periods[i] = best_p
        norms[i] = best_norm
        bases[i] = best_base

        # now get the residual by doing a bunch of stuff
        if i==0:
          # if it's the first time, just subtract and do it again
          res = res - best_base
        else:
          # print ('##########################################')
          # otherwise, we have to do a bunch of stuff
          basis_matricies = [] # a list to later be changed into a tuple
          nonzero_periods = periods[periods>0] # periods that are not 0

          ##########################
          """
          First, find the subspaces that are the GCD's of all of the periods found
          so far. Remove rows as necessary.
          """
          ##########################
          these_gcds = self.get_gcds(list(itertools.combinations(nonzero_periods, 2))) # return a sorted and duplicate-less list of GCDs
          gcd_matricies = [np.ones((1,N))] # space to put the subspaces
          gcd_added = [1] # remember what we've added (same index as gcd_matricies)
          for gcd in these_gcds:
            gcd = int(gcd)
            if gcd == 1:
              pass # already have it
            else:
              # if it's any number other than 1, we're gonna chop some rows. Figure out how many.
              rows_to_chop = 0 # we're gonna chop
              gcd_gcds = self.get_gcds(gcd, gcd_added) # get the gcds between this one and all previously added
              for ii,g in enumerate(gcd_gcds):
                if g in gcd_added:
                  # if this gcd is in the list we've already made, count
                  rows_to_chop += gcd_matricies[ii].shape[0] # count the rows
              gcd_added.append(gcd) # remember this one
              gcd_matricies.append(self.Pt_complete(gcd, N, rows_to_chop)) # make the subspace and append

          gcd_added = np.array(gcd_added) # make it a numpy array for better functionality

          ##########################
          """
          Now that we have our subspaces which are GCDs of each found period, make
          the subspaces for each found period, removing the rows necessary.
          """
          ##########################
          for p in nonzero_periods:
            facs = get_factors(p, remove_1_and_n=True) # get all the factors, removing 1 and n
            rows_to_chop = 1 # start with one since we removed 1 from the factors
            for f in facs:
              try:
                idx = np.where(gcd_added == f)[0][0] # get where this factor occurs
                rows_to_chop += gcd_matricies[idx].shape[0] # count the rows of that factor
              except IndexError:
                pass
            basis_matricies.append(self.Pt_complete(p, N, rows_to_chop)) # add it to the list

          ##########################
          """
          Now we have all of the basis subspaces AND their factors, correctly shaped.
          Find the best representation of the signal using the non-zero periods and
          subtract that from the original to get the new residual.
          """
          ##########################
          res = self._data - self.solve_quadratic(self._data, basis_matricies, gcd_matricies) # get the new residual and do it again
      else:
        break # rms is too low, get out

    output_bases = {'periods': nonzero_periods, 'norms': norms[:len(nonzero_periods)], 'subspaces': basis_matricies}
    output_gcds = {'gcds': gcd_added, 'subspaces': gcd_matricies}
    return (output_bases, output_gcds)

  @staticmethod
  def solve_quadratic(x, bases, gcds):
    A = np.vstack(tuple([*bases, *gcds])) # stack it
    A_prime = np.matmul(A, A.T) # multiply by its transpose
    W = np.matmul(A, x) # multiply by the data
    output = np.linalg.solve(A_prime, W) # actually solve it
    reconstructed = np.matmul(A.T, output) # reconstruct the output
    return reconstructed

  @staticmethod
  def get_gcds(input1, input2=None):
    if isinstance(input1, int):
      gcds = []
      for value in input2:
        gcds.append(np.gcd(input1, value))
      gcds = np.array(gcds)
    else:
      gcds = np.array([np.gcd(l[0],l[1]) for l in input1])

    gcds = np.unique(np.array([gcds])) # remove duplicates
    gcds = np.sort(gcds) # sort ascending
    return gcds

  def Pt_complete(self, p, N=1, lop_off=None):
    repetitions = int(np.ceil(N/p))
    matrix = np.zeros((p, int(N)))
    for i in range(p):
      matrix[i] = self.Pp(p, i, repetitions)[:int(N)]

    if lop_off:
      matrix = matrix[:-lop_off]
    return matrix

  @staticmethod
  def Pp(p, s, repetitions=1):
    vec = np.zeros(p)
    for i in range(p):
      if (i-s)%p == 0:
        vec[i] = 1
    return np.tile(vec, repetitions)


  ######################
  # Properties ########
  ######################

  def data():
      doc = "The data property."
      def fget(self):
          return self._data
      def fset(self, value):
          self._data = value
      return locals()
  data = property(**data())

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
