# pyPeriod

Sethares and Staley's Periodicity Transforms in Python. See the paper [here](https://sethares.engr.wisc.edu/paperspdf/pertrans.pdf).

There is still much work to be done on these in terms of efficiency but it is a faithful implementation of the ideas and algorithms found in the paper. This is an alpha version so expect breaking changes in the future.

## Usage
```
import numpy as np
from pyPeriod import Periods
from random import uniform

# define a signal
sr = 1000 # samplerate
f1 = 10 # frequency
f2 = 17
noise = 0.2 # percent noise

# Make some signals with some noise. The second signal has a slightly offset phase.
a = [np.sin((x*np.pi*2*f1)/sr)+(uniform(-1*noise, noise)) for x in range(2000)]
b = [np.sin(((x*np.pi*2*f2)/sr) + (np.pi*1.1))+(uniform(-1*noise, noise)) for x in range(2000)]
c = np.array(a)+np.array(b) # combine the signals

# normalize, though the algorithms can handle any range of signals (best if DC is removed)
c /= np.max(np.abs(c),axis=0)

p = Periods(c) # make an instance of Period

"""
The output of each algorithm are three arrays consisting of:
  periods: the integer values of the periods found
  powers: the floating point values of the amount of "energy" removed by each period found
  bases: the arrays, equal in length to the input signal length, that contain the periodicities found
"""

# find periodicities using the small-to-large algorithm
periods, powers, bases = p.small_to_large(thresh=0.1)

# find periodicities using the M-best algorithm
periods, powers, bases = p.m_best(num=10)

# find periodicities using the M-best gamma algorithm
periods, powers, bases = p.m_best_gamma(num=10)

# find periodicities using the best correlation algorithm
periods, powers, bases = p.best_correlation(num=10)

# find periodicities using the best frequency algorithm
periods, powers, bases = p.best_frequency(sr=sr, win_size=None, num=10)
## note that for best frequency, we need a samplerate. The window size, if not provided,
## is the same length as the input signal. A larger window is zero-padded, a smaller
## window is truncated (not good).

```

### Algorithms
- Small-to-large
- M-best
- M-best gamma
- Best correlation
- Best frequency (not yet implemented)

### Documentation
Forthcoming. Between the source and the paper, it should be pretty easy to understand in the meantime.

## Requirements
- `numpy>=1.19.2`


## Contributors
- Jacob Sundstrom: University of California, San Diego
