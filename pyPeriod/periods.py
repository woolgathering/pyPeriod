"""
Periodicity transform as described by Sethares and Staley in \"Periodicity  Transforms\",
IEEE  Transactions  on Signal  Processing, Vol. 47, No. 11, November 1999
(https://sethares.engr.wisc.edu/paperspdf/pertrans.pdf)
"""

import numpy as np
from functools import reduce
import math
from warnings import warn

def interleave(x):
  a = np.empty(0)
  for i in range(len(x[0])): # we assume that the first sublist is also the longest
    for j in range(len(x)):
      try:
        a = np.append(a, x[j][i])
      except IndexError:
        pass # move on
  return a

def get_factors(n):
  facs = set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
  # get rid of 1 and the number itself
  facs.remove(1)
  facs.remove(n)
  return facs

def get_primes(max=1000000):
  primes=np.arange(3,max+1,2)
  isprime=np.ones((max-1)/2,dtype=bool)
  for factor in primes[:int(math.sqrt(max))]:
    if isprime[(factor-2)/2]: isprime[(factor*3-2)/2::factor]=0
  return np.insert(primes[isprime],0,2)

class Periods:
  @classmethod
  def inverse(powers, bases):
    """
    A quasi-inverse method to rebuild the original signal. Not theoretically
    sound, use at your own risk.
    """
    pass

  def __init__(self, data):
    super(Periods, self).__init__()
    self._data = data

  def project(self, data, p=2):
    if p==1:
      return data
    else:
      N = len(data)
      projection = np.empty(N)
      fac = 1/math.floor(N/p)
      for s in range(p):
        this_sum = 0
        count = 0
        for n in range(math.ceil(N/p)):
          try:
            this_sum += data[s+(n*p)]
            count += 1
          except IndexError:
            pass # ignore it and move on
            # break # ignore it and move on
        val = this_sum/count # get the average

        # repopulate immediately so we don't have to interleave later (save a loop)
        for n in range(math.ceil(N/p)):
          try:
            projection[s+(n*p)] = val
          except IndexError:
            pass # ignore it and move on
            # break # ignore it and move on

      return projection # return the project


  def __periodic_norm(self, x, p=None):
    # extra arg there to make m_best easier. Can be setup cleaner later.
    return np.linalg.norm(x) / np.sqrt(len(x))

  def small_to_large(self, thresh=0.1, n_periods=None):
    periods = []
    powers = []
    bases = []
    data_norm = self.__periodic_norm(self.data)
    residual = self.data.copy()
    if n_periods is None:
      n_periods = math.floor(len(self.data)/2)
    for p in range(2, n_periods+1):
      base = self.project(self.data, p) # project
      this_residual = residual - base # get the residual
      imposed_norm = (self.__periodic_norm(residual) - self.__periodic_norm(this_residual)) / data_norm
      if imposed_norm > thresh:
        # save it
        residual = this_residual
        periods.append(p)
        powers.append(imposed_norm)
        bases.append(base)
    return (periods, powers, bases)

  def m_best(self, num=5, max_length=None):
    return self.__m_best_meta(self.__periodic_norm, num, max_length)

  def m_best_gamma(self, num=5, max_length=None):
    return self.__m_best_meta(self.__periodic_norm_sqrt, num, max_length)

  def __m_best_meta(self, func_name, num=5, max_length=None):
    if max_length is None:
      max_length = math.floor(len(self.data)/3)
    data_copy = self.data.copy()
    periods = np.zeros(num, dtype=np.uint32)
    norms = np.zeros(num)
    bases = np.zeros((num, len(self.data)))

    # step 1
    for i in range(num):
      max_norm = 0
      max_period = 0
      max_base = None
      for p in range(2,max_length):
        # rewrite this. It's REALLY redundant and can be made a lot faster
        base = self.project(data_copy, p)
        p_norm = func_name(base, p)
        if p_norm > max_norm:
          max_period = p
          max_norm = p_norm
          max_base = base
      periods[i] = max_period
      norms[i] = max_norm
      bases[i] = max_base
      data_copy = data_copy - max_base # remove the best one and do it again

    # step 2
    print ('Now we\'re on step 2')
    changed = True
    while changed:
      i = 0
      while i<num:
        changed = False
        max_norm = 0
        max_period = None
        max_base = None
        facs = get_factors(periods[i])
        for f in facs:
          base = self.project(bases[i], f)
          norm = func_name(base, p)
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
          nq = func_name(xq, p)
          min_q = min(norms)
          if (nq+nQ) > (norms[num-1]+norms[i]) and (nq>min_q) and (nQ>min_q):
            changed = True

            # print ('Changed!')
            # status = '\
            #   i = {} \n\
            #   {} replaced {} in periods \n\
            #   {} replaced {} in norms \n\
            # '.format(i, max_period, periods[i], nq, max_norm)
            # print (status)

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

    powers = norms/self.__periodic_norm(self.data)
    return (periods, powers, bases)

  # for m best gamma
  def __periodic_norm_sqrt(self, x, p):
    return self.__periodic_norm(x) / math.sqrt(p)


  def best_correlation(self, num=5, max_length=None, ratio=0.01):
    if max_length is None:
      max_length = math.floor(len(self._data)/3)
    periods = np.zeros(num, dtype=np.uint32)
    norms = np.zeros(num)
    bases = np.zeros((num, len(self._data)))
    og_norm = self.__periodic_norm(self._data) # original gangsta norm
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
      base = self.project(data_copy, max_period)
      data_copy = data_copy - base
      this_norm = self.__periodic_norm(data_copy)
      norm_test = ((old_norm - this_norm) / og_norm)
      if norm_test > ratio:
        periods[i] = max_period
        norms[i] = norm_test
        bases[i] = base
        old_norm = this_norm

    return (periods, norms, bases)

  def best_frequency(self, sr, win_size=None, num=5):
    if win_size is None:
      win_size = len(self._data)
    elif win_size < len(self._data):
      warn('WARNING: win_size is smaller than the input signal length. It will be truncated and information will be lost')
    if sr is None:
      raise Error('Samplerate (sr) must be set!')

    periods = np.zeros(num, dtype=np.uint32)
    norms = np.zeros(num)
    bases = np.zeros((num, len(self._data)))
    data_copy = self._data.copy()

    for i in range(num):
      mags = np.abs(np.fft.rfft(data_copy, win_size)) # we only need the magnitude of the positive freqs
      p = ((sr*0.5)/win_size) * np.argmax(mags) # get the frequency
      p = int(np.round(sr/p)) # convert it to a period and round
      base = self.project(data_copy, p) # project
      # remember it
      periods[i] = p
      norms[i] = self.__periodic_norm(base)
      bases[i] = base
      data_copy = data_copy - base # remove it

    powers = norms / self.__periodic_norm(self._data)
    return (periods, powers, bases)


  ######################
  # Properties ########
  ######################

  def data():
      doc = "The data property."
      def fget(self):
          return self._data
      def fset(self, value):
          self._data = value
      def fdel(self):
          del self._data
      return locals()
  data = property(**data())
