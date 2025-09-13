import unittest
import pandas as pd
import numpy as np
from group_stats import calculate_group_stats

class TestGroupStats(unittest.TestCase):

    def setUp(self):
        """Set up a dummy dataframe for testing."""
        data = {
            'group': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
            'experiment_id': [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
            'koff': [0.1, 0.12, 0.11, 0.13, 0.2, 0.22, 0.21, 0.23, 0.3, 0.32, 0.31, 0.33],
            'D_um2_s': [1.0, 1.2, 1.1, 1.3, 2.0, 2.2, 2.1, 2.3, 3.0, 3.2, 3.1, 3.3],
            'app_mw_kDa': [10, 12, 11, 13, 20, 22, 21, 23, 30, 32, 31, 33],
            'mobile_fraction': [50, 55, 52, 58, 60, 65, 62, 68, 70, 75, 72, 78]
        }
        self.df = pd.DataFrame(data)
        self.metrics = ['koff', 'D_um2_s', 'app_mw_kDa', 'mobile_fraction']

    def test_two_groups(self):
        """Test with two groups."""
        df_two_groups = self.df[self.df['group'].isin(['A', 'B'])].copy()
        results = calculate_group_stats(
            data=df_two_groups,
            metrics=self.metrics,
            group_order=['A', 'B']
        )
        self.assertIsInstance(results, pd.DataFrame)
        self.assertFalse(results.empty)
        self.assertEqual(len(results), len(self.metrics))
        self.assertIn('q_value', results.columns)

    def test_three_groups(self):
        """Test with three groups."""
        results = calculate_group_stats(
            data=self.df,
            metrics=self.metrics,
            group_order=['A', 'B', 'C']
        )
        self.assertIsInstance(results, pd.DataFrame)
        self.assertFalse(results.empty)
        # ANOVA result + 3 pairwise comparisons for each metric
        self.assertEqual(len(results), len(self.metrics) * 4)
        self.assertIn('q_value', results.columns)

    def test_tost(self):
        """Test TOST equivalence."""
        df_two_groups = self.df[self.df['group'].isin(['A', 'B'])].copy()
        tost_thresholds = {'D_um2_s': (-0.2, 0.2)}
        results = calculate_group_stats(
            data=df_two_groups,
            metrics=['D_um2_s'],
            group_order=['A', 'B'],
            tost_thresholds=tost_thresholds
        )
        self.assertIn('tost_outcome', results.columns)
        self.assertNotEqual(results['tost_outcome'].iloc[0], "N/A")

    def test_mixed_effects(self):
        """Test mixed effects model."""
        # This is a bit harder to assert, so we'll just check if it runs without error
        # and returns a summary. The output is printed, so we can't capture it easily here.
        # A more advanced test would redirect stdout.
        try:
            calculate_group_stats(
                data=self.df,
                metrics=['koff'],
                group_order=['A', 'B', 'C'],
                use_mixed_effects=True,
                random_effect_col='experiment_id'
            )
        except Exception as e:
            self.fail(f"Mixed effects model test failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()
