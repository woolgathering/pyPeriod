import os
import sys
import numpy as np
from pyPeriod import Periods, QOPeriods

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyPeriod

N = 1000
period = 10
t = np.arange(N)
x = [np.sin(2 * np.pi * t / period)]

periods = Periods(trunc_to_integer_multiple = False, orthogonalize = False)