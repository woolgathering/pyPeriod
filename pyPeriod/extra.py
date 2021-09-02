"""
Extra functions that are not in the paper. Mostly my own experiments.
"""

import math
import numpy as np
from scipy.signal import butter, filtfilt
from pyPeriod import Periods
from random import randint
import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
import multiprocessing as mp
import traceback
from time import time
import datetime


def stretch_by_proportion_phase_bash(periods,
                                     powers,
                                     bases,
                                     proportion=1,
                                     return_inverse=False):
    """
  Stretch by proportion. Values of proportion less than 1 will truncate the signal.
  This is accomplished by calculating the new duration in samples, then adding
  extra periods in each base until the new duration is attained.

  If return_inverse is true, the quasi-inverted periodicity transform is returned.
  Otherwise, the newly extended bases are returned. All other values are the same.
  """

    this_len = bases[0].size
    new_len = int(math.floor(this_len * proportion))
    add_to_len = new_len - this_len

    print([new_len, add_to_len])

    if proportion == 1:
        print('Proportion is 1. Nothing changed.')
        return (periods, powers, bases)
    if new_len < this_len:
        print('Proportion is less than 1: signal will be trunctaed')
        return (periods, powers, bases[:new_len])
    else:
        new_bases = bases.copy()
        new_bases = np.zeros((len(bases), new_len))
        # for i,b in enumerate(new_bases):
        #   new_bases[i] = np.append(bases[i], np.zeros(add_to_len))

    for i, p in enumerate(periods):
        # don't do "empty" basis elements (p=0). Happens with best correlation.
        if p != 0:
            # for each period, check to see what number of samples at the end
            # is a fraction of the period. Then start there.
            mod = this_len % p  # make it negative so we can just index directly

            zero_phase_len = (len(new_bases[i]) + (p - len(new_bases[i]) % p))
            tmp_base = np.append(bases[i],
                                 np.zeros(zero_phase_len - len(bases[i])))

            for j in range(mod + (zero_phase_len - len(bases[i]))):
                # j is the reading index, k is the writing index
                # k = j-mod-add_to_len
                k = j - mod - (zero_phase_len - len(bases[i]))
                tmp_base[k] = tmp_base[j]

            tmp_base = np.roll(tmp_base, randint(0, p))  # random phase shift
            new_bases[i] = tmp_base[:new_len]
        else:
            pass

    if return_inverse:
        return sum(new_bases)
    else:
        return new_bases


def stretch_by_proportion(periods,
                          powers,
                          bases,
                          proportion=1,
                          return_inverse=False):
    """
  Stretch by proportion. Values of proportion less than 1 will truncate the signal.
  This is accomplished by calculating the new duration in samples, then adding
  extra periods in each base until the new duration is attained.

  If return_inverse is true, the quasi-inverted periodicity transform is returned.
  Otherwise, the newly extended bases are returned. All other values are the same.
  """

    this_len = bases[0].size
    new_len = int(math.floor(this_len * proportion))
    add_to_len = new_len - this_len

    if proportion == 1:
        print('Proportion is 1. Nothing changed.')
        return (periods, powers, bases)
    if new_len < this_len:
        print('Proportion is less than 1: signal will be trunctaed')
        return (periods, powers, bases[:new_len])
    else:
        new_bases = bases.copy()
        new_bases = np.zeros((len(bases), new_len))
        for i, b in enumerate(new_bases):
            new_bases[i] = np.append(bases[i], np.zeros(add_to_len))

    for i, p in enumerate(periods):
        # don't do "empty" basis elements (p=0). Happens with best correlation.
        if p != 0:
            # for each period, check to see what number of samples at the end
            # is a fraction of the period. Then start there.
            mod = this_len % p  # make it negative so we can just index directly
            for j in range(mod + add_to_len):
                # j is the reading index, k is the writing index
                k = j - mod - add_to_len
                new_bases[i][k] = new_bases[i][j]

        else:
            pass

    if return_inverse:
        return sum(new_bases)
    else:
        return new_bases


def stretch_by_samples(periods,
                       powers,
                       bases,
                       samples=0,
                       return_inverse=False):
    """
  Stretch by samples. Give a number of samples to stretch by.
  This is accomplished by calculating the new duration in samples, then
  adding extra periods in each base until the new duration is attained.

  If return_inverse is true, the quasi-inverted periodicity transform is returned.
  Otherwise, the newly extended bases are returned. All other values are the same.
  """
    pass


def stretch_by_chunk(x,
                     stretch_prop=1,
                     chunk_size=1024,
                     overlap=0.1,
                     algorithm='m_best',
                     **kwargs):
    """
  A way (???) to stretch a file incrementally.

  "Chunks" of the file are taken to be windows (a la FFT). They are then stretched by the
  proportion incrementally in one chunk at a time. Who knows if this will work.
  """
    # edge_prop = overlap*2
    # samp_overlap = int(chunk_size*edge_prop*0.5) # number of samples to overlap per chunk
    advance = int(chunk_size * (1 - overlap))
    num_chunks = math.floor(len(x) / advance)  # just truncate for now
    old_chunk = None
    all_periods = []
    all_powers = []

    print(num_chunks)

    for i in range(num_chunks):
        print(i)
        start_idx = i * advance
        chunk = x[start_idx:start_idx + chunk_size]
        this_period = Periods(chunk)

        this_algorithm = getattr(this_period, algorithm)
        periods, powers, bases = this_algorithm(**kwargs)

        all_periods.append(periods)
        all_powers.append(powers)
        chunks = stretch_by_proportion(periods, powers, bases, stretch_prop)
        chunk = sum(chunks)

        # filter?
        # b, a = butter(30, 15000/(44100*0.5), btype='low', analog=False)
        # chunk = filtfilt(b, a, chunk)

        samp_overlap = int(len(chunk) * overlap)
        win = equal_power_win(len(chunk), overlap)  # make the window now
        # win = linear_win(len(chunk), overlap) # make the window now
        # print (len(win))
        chunk = chunk * win

        if i == 0:
            old_chunk = chunk
        else:
            if (i % 2) == 1:
                chunk = chunk * -1  # invert the phase of every other window
            old_chunk = overlap_add(old_chunk, chunk, samp_overlap)

    # return old_chunk
    return (old_chunk, np.array(all_periods), np.array(all_powers))


# struct is iterations by mbest number (i.e. 5x10 are 5 nested m-best where m=10)
def resynthesize_mbest(x,
                       window_size=1024,
                       min_length=2,
                       struct=[5, 10],
                       overlap=0,
                       min_cont_pow=0,
                       trunc_to_integer_multiple=False,
                       orthogonalize=False,
                       verbose=False):
    # no overlap
    overlap_samples = int(window_size * overlap)
    try:
        amp_inc = 1 / overlap_samples
    except ZeroDivisionError:
        amp_inc = 1
    advance = window_size - overlap_samples
    windows = int(np.round(len(x) / advance))
    output = np.zeros(len(x), dtype=np.float128)
    all_periods = []
    exec_time = time()

    print('Number of windows: {}'.format(windows))

    if verbose:
        print(' \
      Overlap Samples: {}\n \
      Advance: {}\n \
      Amplitude Increment: {}\n \
    '.format(overlap_samples, advance, amp_inc))

    for i in range(windows):
        print('Processing window {}'.format(i + 1))
        # try:
        start = i * advance
        end = start + window_size
        sig = x[start:min(start + window_size, len(x) - 1)]
        sig = remove_dc(sig)
        window_out = np.zeros(len(sig), np.float128)
        sig_pow = np.sqrt(np.sum(np.power(sig, 2)))
        n = struct[0]
        these_periods = []

        if verbose:
            print(' \
      \tStart: {} \
      \tEnd: {} \
      '.format(start, end))

        try:
            while n > 0:
                n -= 1
                p = Periods(sig, trunc_to_integer_multiple, orthogonalize)
                periods, powers, bases = p.m_best_gamma(
                    struct[1], None, min_length)
                # at this point, only keep bases that meet a certain threshold of power
                # powers = powers/sig_pow
                idxs = np.where(powers > min_cont_pow)[0]
                # bases = bases[idxs]
                reconstruction = np.sum(bases, 0)  # naively invert
                residual = sig - reconstruction
                sig = residual
                window_out += reconstruction
                window_out = remove_dc(window_out)
                these_periods.append(np.array([periods[idxs], powers[idxs]
                                               ]))  # remember this iteration

                if verbose:
                    print(' \
          \tIteration: {} \n\
          \tPeriods: {} \n\
          \tPowers: {} \n\
          '.format(n + 1, periods, powers))

            all_periods.append(these_periods)  # remember this window

            if i == 0:
                # if it's the first window, just taper the end
                for a in range(overlap_samples):
                    amp = max(1 - (amp_inc * a), 0)
                    window_out[-1 *
                               (overlap_samples -
                                a)] = window_out[-1 *
                                                 (overlap_samples - a)] * amp
                # add it to the output array
                for num, samp in enumerate(window_out):
                    output[num] = samp
                # output[start:min(start+window_size, len(x)-1)] = window_out
            elif i == (windows - 1):
                # otherwise it's the last sample so just taper the beginning
                for a in range(overlap_samples):
                    amp = min(amp_inc * a, 1)
                    window_out[a] = window_out[a] * amp
                # output[end-overlap_samples] += window_out
                for num, samp in enumerate(window_out):
                    output[num + start] += samp
            else:
                # else it's netiher so do both
                for a in range(overlap_samples):
                    amp_up = min(amp_inc * a, 1)
                    amp_down = 1 - amp_up
                    window_out[a] = window_out[a] * amp_up
                    window_out[-1 * (overlap_samples - a)] = window_out[
                        -1 * (overlap_samples - a)] * amp_down
                # output = overlap_add(output, window_out, overlap_samples)
                # output[end-overlap_samples:] += window_out
                for num, samp in enumerate(window_out):
                    output[num + start] += samp
        except TypeError as err:
            print('TypeError, problem in m-best!!')
            print(traceback.format_exc())
            print(err)
            pass

    exec_time = datetime.timedelta(seconds=time() - exec_time)
    print('Execution time: {}'.format(exec_time))

    return [output, all_periods]


# struct is iterations by mbest number (i.e. 5x10 are 5 nested m-best where m=10)
def resynthesize_quadopt(x,
                         window_size=1024,
                         thresh=0.05,
                         max_periods_to_find=15,
                         max_length=None,
                         overlap=0,
                         min_cont_pow=0,
                         trunc_to_integer_multiple=False,
                         old=False,
                         verbose=False):
    def centroid(mags, freqs):
        return np.sum(mags * freqs) / np.sum(mags)

    overlap_samples = int(window_size * overlap)
    try:
        amp_inc = 1 / overlap_samples
    except ZeroDivisionError:
        amp_inc = 1
    advance = window_size - overlap_samples
    windows = int(np.round(len(x) / advance)) - 1
    output = np.zeros(len(x), dtype=np.float64)
    all_periods = []
    exec_time = time()
    win = np.hamming(window_size)

    print('Number of windows: {}'.format(windows))

    if verbose:
        print(' \
      Overlap Samples: {}\n \
      Advance: {}\n \
      Amplitude Increment: {}\n \
    '.format(overlap_samples, advance, amp_inc))

    for i in range(windows):
        if verbose:
            print('>>>>>>>>>>>>>>>>>>>>>>>>\n>>>>>>>>>>>>>>>>>>>>>>>>')
        print('Processing window {}/{}'.format(i + 1, windows))
        # try:
        start = i * advance
        end = min(start + window_size, len(x) - 1)
        sig = x[start:end]
        sig = remove_dc(sig)
        if len(sig) < window_size:
            sig = np.pad(sig, (0, window_size - len(sig)))
        window_out = np.zeros(len(sig), np.float64)
        these_periods = {}

        if verbose:
            print(' \
      \tStart: {} \
      \tEnd: {} \
      '.format(start, end))

        try:
            p = Periods(sig, trunc_to_integer_multiple)
            # spec = np.abs(np.fft.rfft(sig*win))
            # freqs = np.arange(len(spec))
            # center = centroid(spec/(np.arange(1,len(spec)+1)), freqs)
            # min_length = int(np.ceil(len(sig)/center))
            bases, gcds = p.quadratic_optimization(max_periods_to_find, thresh,
                                                   8, max_length, old)
            these_periods['periods'] = bases['periods']
            these_periods['norms'] = bases['norms']
            window_out = p.solve_quadratic(sig, bases['subspaces'],
                                           gcds['subspaces'])

            if verbose:
                print('\tPeriods found: {}'.format(these_periods))

            all_periods.append(these_periods)  # remember this window

            if i == 0:
                # if it's the first window, just taper the end
                for a in range(overlap_samples):
                    amp = max(1 - (amp_inc * a), 0)
                    window_out[-1 *
                               (overlap_samples -
                                a)] = window_out[-1 *
                                                 (overlap_samples - a)] * amp
                # add it to the output array
                for num, samp in enumerate(window_out):
                    output[num] = samp
                # output[start:min(start+window_size, len(x)-1)] = window_out
            elif i == (windows - 1):
                # otherwise it's the last sample so just taper the beginning
                for a in range(overlap_samples):
                    amp = min(amp_inc * a, 1)
                    window_out[a] = window_out[a] * amp
                # output[end-overlap_samples] += window_out
                for num, samp in enumerate(window_out):
                    output[num + start] += samp
            else:
                # else it's netiher so do both
                for a in range(overlap_samples):
                    amp_up = min(amp_inc * a, 1)
                    amp_down = 1 - amp_up
                    window_out[a] = window_out[a] * amp_up
                    window_out[-1 * (overlap_samples - a)] = window_out[
                        -1 * (overlap_samples - a)] * amp_down
                # output = overlap_add(output, window_out, overlap_samples)
                # output[end-overlap_samples:] += window_out
                for num, samp in enumerate(window_out):
                    output[num + start] += samp
        except np.linalg.LinAlgError as err:
            print('LinAlgError, singular matrix?')
            print(traceback.format_exc())
            print(err)
            pass

    exec_time = datetime.timedelta(seconds=time() - exec_time)
    print('Execution time: {}'.format(exec_time))

    return [output, all_periods]


def remove_dc(a):
    avg = sum(a) / len(a)
    return a - avg


def prep_periodogram(frames):
    # pass
    # frames is an array of depth-2 arrays: [[f1_pers, f1_pows], [f2_pers, f2_pows]...]
    # ugly but works
    minmax_pers = [0, 0]
    for f in frames:
        # keep track of the biggest and smallest periods
        if min(f[0]) > minmax_pers[0]:
            minmax_pers[0] = min(f[0])
        if max(f[0]) > minmax_pers[1]:
            minmax_pers[1] = max(f[0])

    X = []
    for f in frames:
        im_frame = np.zeros(int(np.diff(minmax_pers)[0]))
        for i, per in enumerate(f[0]):
            im_frame[int(per - minmax_pers[0])] = f[1][i]
        X.append(im_frame)
    X = np.array(X)
    return (X.T, minmax_pers)


#########################
## Helper functions
#########################
def equal_power_win(length=1024, edge_prop=0.1):
    angle = np.linspace(0, np.pi * 0.5,
                        math.floor(edge_prop * length))  # angle to get the pan
    att = np.sin(angle)
    rel = np.cos(angle)
    sus_len = length - (len(att) + len(rel))
    sus = np.linspace(1, 1, sus_len)
    return np.append(np.append(att, sus), rel)


def linear_win(length=1024, edge_prop=0.1):
    att = np.linspace(0, 1,
                      math.floor(edge_prop * length))  # angle to get the pan
    rel = np.linspace(1, 0,
                      math.floor(edge_prop * length))  # angle to get the pan
    sus_len = length - (len(att) + len(rel))
    sus = np.linspace(1, 1, sus_len)
    return np.append(np.append(att, sus), rel)


def overlap_add(a, b, overlap_samps=0):
    new_size = len(a) + len(b) - overlap_samps
    tmp = np.zeros(0, 0, new_size)
    tmp[:len(a)] = a
    tmp[len(a) - overlap_samps:] += b
    return tmp


# add b to a starting at start_sample
# def overlap_add(a, b, start_sample=0):
#   for i in range(len(b)):
#     a[start_sample+i] = a[start_sample+i] + b[i]
#   return a


def process_window(i, x, window_size, min_length, iterations, overlap_samples,
                   advance, output, all_periods):
    print('Processing window {}'.format(i))
    # try:
    start = i * advance
    end = start + window_size
    sig = x[start:min(start + window_size, len(x) - 1)]
    sig = remove_dc(sig)
    window_out = np.zeros(len(sig), np.float128)
    sig_pow = np.sqrt(np.sum(np.power(sig, 2)))
    n = iterations
    these_periods = []
    try:
        while n > 0:
            n -= 1
            # p = Periods(sig, np.float128)
            p = Periods(sig)
            periods, powers, bases = p.m_best_gamma(10, None, min_length)
            # at this point, only keep bases that meet a certain threshold of power
            powers = powers / sig_pow
            idxs = np.where(powers > min_cont_pow)[0]
            these_periods.append(np.array([periods[idxs], powers[idxs]]))

            reconstruction = sum(bases)  # naively invert
            residual = sig - reconstruction
            sig = residual
            window_out += reconstruction
            window_out = remove_dc(window_out)

        all_periods.append(these_periods)

        if i == 0:
            # if it's the first window, just taper the end
            for a in range(overlap_samples):
                amp = max(1 - (amp_inc * a), 0)
                window_out[-1 * (overlap_samples -
                                 a)] = window_out[-1 *
                                                  (overlap_samples - a)] * amp
            # output[start] += window_out
            # add it to the output array
            for num, samp in enumerate(window_out):
                output[num] = samp
        elif i == (overlap_samples - 1):
            # otherwise it's the last sample so just taper the beginning
            for a in range(overlap_samples):
                amp = min(amp_inc * a, 1)
                window_out[a] = window_out[a] * amp
            # output[start] += window_out
            for num, samp in enumerate(window_out):
                output[num + start] += samp
        else:
            # else it's netiher so do both
            for a in range(overlap_samples):
                amp_up = min(amp_inc * a, 1)
                amp_down = 1 - amp_up
                window_out[a] = window_out[a] * amp_up
                window_out[-1 *
                           (overlap_samples -
                            a)] = window_out[-1 *
                                             (overlap_samples - a)] * amp_down
            # output[start] += window_out
            for num, samp in enumerate(window_out):
                output[num + start] += samp
    except TypeError:
        print('TypeError, problem in m-best!!')
        pass


# end process_window


def resynthesize_parallel(x,
                          window_size=1024,
                          min_length=2,
                          iterations=5,
                          overlap=0,
                          min_cont_pow=0,
                          processes=None):
    # no overlap
    overlap_samples = int(window_size * overlap)
    amp_inc = 1 / overlap_samples
    advance = window_size - overlap_samples
    windows = math.floor(len(x) / advance)
    output = np.zeros(len(x), dtype=np.float128)
    all_periods = []
    print('Number of windows: {}'.format(windows))

    # prepare and run the parallel processes
    pool = mp.Pool(processes=processes,
                   initargs=(x, window_size, min_length, iterations,
                             overlap_samples, advance, output, all_periods))
    pool.map(process_window, [i for i in range(windows)])
    pool.terminate()

    return [output, all_periods]


###################################


# struct is iterations by mbest number (i.e. 5x10 are 5 nested m-best where m=10)
def stretch_sample(x,
                   window_size=1024,
                   min_length=2,
                   struct=[5, 10],
                   overlap=0,
                   min_cont_pow=0,
                   stretch=1,
                   trunc_to_integer_multiple=False,
                   orthogonalize=False,
                   verbose=False):
    # no overlap
    overlap_samples = int(window_size * overlap)
    try:
        amp_inc = 1 / overlap_samples
    except ZeroDivisionError:
        amp_inc = 1
    advance = window_size - overlap_samples
    windows = int(np.round(len(x) / advance))
    output = np.zeros(int(np.ceil(len(x) * (stretch + 1))), dtype=np.float128)
    all_periods = []
    exec_time = time()

    print('Number of windows: {}'.format(windows))

    if verbose:
        print(' \
      Overlap Samples: {}\n \
      Advance: {}\n \
      Amplitude Increment: {}\n \
    '.format(overlap_samples, advance, amp_inc))

    for i in range(windows):
        print('Processing window {}'.format(i + 1))
        # try:
        start = i * advance
        end = start + window_size
        sig = x[start:min(start + window_size, len(x) - 1)]
        sig = remove_dc(sig)
        window_out = np.zeros(int(window_size * stretch), np.float128)
        sig_pow = np.sqrt(np.sum(np.power(sig, 2)))
        n = struct[0]
        these_periods = []

        if verbose:
            print(' \
      \tStart: {} \
      \tEnd: {} \
      '.format(start, end))

        try:
            while n > 0:
                n -= 1
                p = Periods(sig, trunc_to_integer_multiple, orthogonalize)
                periods, powers, bases = p.m_best_gamma(
                    struct[1], None, min_length)
                # at this point, only keep bases that meet a certain threshold of power
                # powers = powers/sig_pow
                idxs = np.where(powers > min_cont_pow)[0]
                # bases = bases[idxs]

                stretched_bases = np.zeros(
                    (len(bases), int(window_size * stretch)))
                for ii, b in enumerate(bases):
                    reps = int(np.ceil(
                        (window_size * stretch) / periods[ii])) + 1
                    stretched_bases[ii] = np.tile(
                        b[:periods[ii]], reps)[:int(window_size * stretch)]

                stretched_reconstruction = np.sum(stretched_bases,
                                                  0)  # naively invert
                reconstruction = np.sum(bases, 0)  # naively invert
                residual = sig - reconstruction
                sig = residual
                window_out += np.pad(
                    stretched_reconstruction,
                    (0, len(window_out) - len(stretched_reconstruction)))
                window_out = remove_dc(window_out)
                these_periods.append(np.array([periods[idxs], powers[idxs]
                                               ]))  # remember this iteration

                if verbose:
                    print(' \
          \tIteration: {} \n\
          \tPeriods: {} \n\
          \tPowers: {} \n\
          '.format(n + 1, periods, powers))

            all_periods.append(these_periods)  # remember this window

            this_overlap_samples = int(overlap_samples * stretch)
            this_amp_inc = amp_inc / stretch
            this_start = int(start * stretch)

            if i == 0:
                # if it's the first window, just taper the end
                for a in range(this_overlap_samples):
                    amp = max(1 - (this_amp_inc * a), 0)
                    window_out[-1 * (this_overlap_samples - a)] = window_out[
                        -1 * (this_overlap_samples - a)] * amp
                # add it to the output array
                for num, samp in enumerate(window_out):
                    output[num] = samp
                # output[this_start:min(this_start+window_size, len(x)-1)] = window_out
            elif i == (windows - 1):
                # otherwise it's the last sample so just taper the beginning
                for a in range(this_overlap_samples):
                    amp = min(this_amp_inc * a, 1)
                    window_out[a] = window_out[a] * amp
                # output[end-this_overlap_samples] += window_out
                for num, samp in enumerate(window_out):
                    output[num + this_start] += samp
            else:
                # else it's netiher so do both
                for a in range(this_overlap_samples):
                    amp_up = min(this_amp_inc * a, 1)
                    amp_down = 1 - amp_up
                    window_out[a] = window_out[a] * amp_up
                    window_out[-1 * (this_overlap_samples - a)] = window_out[
                        -1 * (this_overlap_samples - a)] * amp_down
                # output = overlap_add(output, window_out, this_overlap_samples)
                # output[end-this_overlap_samples:] += window_out
                for num, samp in enumerate(window_out):
                    output[num + this_start] += samp
        except TypeError as err:
            print('TypeError, problem in m-best!!')
            print(traceback.format_exc())
            print(err)
            pass

    exec_time = datetime.timedelta(seconds=time() - exec_time)
    print('Execution time: {}'.format(exec_time))

    return [output, all_periods]
