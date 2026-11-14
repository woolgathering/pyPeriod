import unittest
import numpy as np
from pyPeriod import Periods
from pyPeriod.utils import sine_func, normalize, remove_dc

import sys
import os

# Add the parent directory of 'pyPeriod' to sys.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pyPeriod"))
)

class TestPeriods(unittest.TestCase):
    
    def setUp(self):
        """Set up test signals and instance of Periods"""
        # Create instance of Periods
        self.periods = Periods(trunc_to_integer_multiple=True, orthogonalize=False)
        
        # Create composite signals with mutually prime periods 8, 15, 21
        x = np.linspace(0, 1000, 10000)
        self.sine_wave_8 = sine_func(x, fund=8)
        self.sine_wave_15 = sine_func(x, fund=15)
        self.sine_wave_21 = sine_func(x, fund=21)

        # Combine the three sine waves into one composite signal
        self.composite_signal = normalize(self.sine_wave_8 + self.sine_wave_15 + self.sine_wave_21)

        # Define expected input periods
        self.expected_periods = [8, 15, 21]

    def check_detected_periods(self, detected_periods, expected_periods, powers):
        """Helper function to verify that the strongest periods match the input components"""
        # Sort periods by descending power
        sorted_idx = np.argsort(-powers)
        detected_periods_sorted = np.array(detected_periods)[sorted_idx]

        # Check that the strongest periods are the same as the input components
        for expected_period in expected_periods:
            self.assertIn(expected_period, detected_periods_sorted)

    def test_project(self):
        """Test the project method with the composite signal"""
        projected = self.periods.project(self.composite_signal, p=8)
        
        self.assertEqual(len(projected), len(self.composite_signal))

    def test_periodic_norm(self):
        """Test the periodic_norm method"""
        norm = self.periods.periodic_norm(self.composite_signal, p=8)
        
        self.assertIsInstance(norm, float)
        self.assertGreater(norm, 0)

    def test_small_to_large(self):
        """Test small_to_large period detection on composite signal"""
        periods, powers, bases = self.periods.small_to_large(self.composite_signal, thresh=0.1)
        
        self.assertEqual(len(periods), len(powers))
        self.assertGreater(len(periods), 0)
        self.check_detected_periods(periods, self.expected_periods, powers)

    def test_best_correlation(self):
        """Test best_correlation method on composite signal"""
        periods, norms, bases = self.periods.best_correlation(self.composite_signal, num=3)
        
        self.assertEqual(len(periods), 3)
        self.assertEqual(len(norms), 3)
        self.assertEqual(bases.shape, (3, len(self.composite_signal)))
        self.check_detected_periods(periods, self.expected_periods, norms)

    def test_best_frequency(self):
        """Test best_frequency method on composite signal"""
        periods, powers, bases = self.periods.best_frequency(self.composite_signal, win_size=1000, num=3)
        
        self.assertEqual(len(periods), 3)
        self.assertEqual(len(powers), 3)
        self.assertEqual(bases.shape, (3, len(self.composite_signal)))
        self.check_detected_periods(periods, self.expected_periods, powers)

    def test_m_best(self):
        """Test m_best period detection on composite signal"""
        periods, powers, bases = self.periods.m_best(self.composite_signal, num=3)
        
        self.assertEqual(len(periods), 3)
        self.assertEqual(len(powers), 3)
        self.assertEqual(bases.shape, (3, len(self.composite_signal)))
        self.check_detected_periods(periods, self.expected_periods, powers)

    def test_m_best_gamma(self):
        """Test m_best_gamma period detection on composite signal"""
        periods, powers, bases = self.periods.m_best_gamma(self.composite_signal, num=3)
        
        self.assertEqual(len(periods), 3)
        self.assertEqual(len(powers), 3)
        self.assertEqual(bases.shape, (3, len(self.composite_signal)))
        self.check_detected_periods(periods, self.expected_periods, powers)

    def test_bases_accuracy(self):
        """Check if the bases are roughly equal to the input components"""
        _, _, bases = self.periods.m_best(self.composite_signal, num=3)

        # Check if the bases are similar to the sine wave components
        # Use cosine similarity or a correlation metric to ensure accuracy
        for base in bases:
            corr_8 = np.corrcoef(base, self.sine_wave_8)[0, 1]
            corr_15 = np.corrcoef(base, self.sine_wave_15)[0, 1]
            corr_21 = np.corrcoef(base, self.sine_wave_21)[0, 1]
            
            # The correlation should be strong with at least one of the components
            self.assertTrue(corr_8 > 0.8 or corr_15 > 0.8 or corr_21 > 0.8)

if __name__ == '__main__':
    unittest.main()
