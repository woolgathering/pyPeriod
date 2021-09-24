# pyPeriod

This package now includes a few different periodicity transforms and will likley be further updated in the future.

### Periods
Sethares and Staley's Periodicity Transforms in Python. See the paper [here](https://sethares.engr.wisc.edu/paperspdf/pertrans.pdf).

Efficiency was improved by modifying the projection algorithm so that we take advantage of the very fast matrix operations in Numpy. The result is identical. There are two flags in the `project` method: `trunc_to_integer_multiple` and `orthogonalize`.
- `trunc_to_integer_multiple` allows for the input to be truncated to an integer multiple of the period one is testing for. Say `p = 3` and `N = len(x)`, we allow the algorithm to truncate `x` so that `3|N`. Assuming a period is constant in the analysis window (with regard to values), setting this as `True` tends to produce a better result and minimizes the residual more than otherwise.
- `orthogonalize` removes prime factors of `p` from the projection. That is, for some period `p` and its projection `y`, we project `y` onto the prime factors of `p` to get `y_hat`. Then we do `y - y_hat` and return the result. This idea was taken from "Orthogonal, exactly periodic subspace decomposition" (D.D. Muresan, T.W. Parks), 2003, and the process simplified. Setting both `trunc_to_integer_multiple` and `orthogonalize` to `True` results in a projection that is identical to that acquired using the process described in Muresan and Parks. Note that `orthogonalize` has no effect in the M-best family as it makes no sense.

### RamanujanPeriods
Periodic decomposition using Ramanujan summation, taken from the work of P. P. Vaidyanathan and Srikanth V. Tenneti. This class is nowhere near completion (I have parts of the entire thing laying around my machine...) but this does a basic decomposition using the basis vectors derived from the Ramanujan sum.

A method is included in this class (`find_periods_with_weights`) which will return the periods found by the Ramanujan transform above some threshold (`thresh`). Using this with `get_periods()` essentially combines the the Ramanujan transform with the quadratic program of `QOPeriods` and allows for good (albeit slow) results without fidding with parameters as is often the case in `QOPeriods`.

### QOPeriods
This is an adaptation of the Sethares and Staley transform where the residual is computed by framing the problem as a quadratic program and each projection is orthogonalized. This produces a better result (up to a point) and is the basis for my forthcoming PhD dissertation on audio applications of these transforms. Note that because this particular ''residualization'' employs a quadratic program, the results of all previously found periods can and often will change. The criteria of when to stop pursuing further periods in the face of residual noise is an ongoing problem and likley unique to each situation. To this end, `find_periods()` can be passed an argument called `test_function` which in turn is passed the residual and expects a boolean return on whether or not to process again.

A method is also included in this class called `get_periods()` that allows one to retrieve the original periods for further analysis.

EXPECT BREAKING CHANGES IN THE FUTURE AS THESE ARE UPDATED AND CHANGED.

## Installation

__Easiest:__

```
pip install pyPeriod
```

#### Development
__From Github__:

```
pip install git+https://github.com/woolgathering/pyPeriod
```

__Via cloning__:

```
git clone https://github.com/woolgathering/pyPeriod.git
pip install -r pyPeriod/requirements.txt
pip install ./pyPeriod
```

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


##################################
## Periods (Sethares and Staley, Muresan and Parks if orthogonalize and trunc_to_integer_multiple are True)
##################################
p = Periods(c) # make an instance of Period (Sethares and Staley)
# p = Periods(c, trunc_to_integer_multiple=True, orthogonalize=True) # make an instance of Period (Muresan and Parks)

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
periods, powers, bases = p.best_frequency(win_size=None, num=10)
## The window size, if not provided, is the same length as the input signal. A
## larger window is zero-padded, a smaller window is truncated (not good).


##################################
## QOPeriods
##################################

# biggest different between this and Periods is that there is no need to pass
# the data at construction
p = QOPeriod() # make an instance

# output is a tuple of a dictionary that contains all the information needed
# to reconstruct the input signal
output, residual = p.find_periods(c, num=2, thresh=0.05)

# get the original periods back as a tuple
isolated_periods = period.get_periods(output['weights'], output['basis_dictionary'])

##################################
## RamanujanPeriods
##################################

# biggest different between this and Periods is that there is no need to pass
# the data at construction
p = RamanujanPeriods() # make an instance

# output is a tuple of a dictionary that contains all the information needed
# to reconstruct the input signal
output, residual = p.find_periods(c, thresh=0.2) # threshold is a fraction of the max energy in the periodogram

# get the original periods back as a tuple
isolated_periods = period.get_periods(output['weights'], output['basis_dictionary'])
```

### Algorithms
#### Only for Periods (i.e. Setheres and Staley (* indicates Muresan and Parks is possible))
- Small-to-large*
- M-best
- M-best gamma
- Best correlation*
- Best frequency*

### Documentation
Forthcoming. Between the source and the paper, it should be pretty easy to understand in the meantime.

## Requirements
- `numpy>=1.19.2`
- `scipy>=1.7.1`

## Update Log
### 0.2.0
    - Added two classes (`QOPeriods` and `RamanujanPeriods`).
    - Removed samplerate requirement for `Periods.best_frequency()`.

## Contributors
- Jacob Sundstrom: University of California, San Diego
