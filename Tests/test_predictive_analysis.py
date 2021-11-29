import unittest
import pandas as pd
import predictive_analysis as pa

test_stock_data = pd.read_csv('amazon_test_data.csv', index_col=0)
test_stock_data.index = pd.to_datetime(test_stock_data.index).date


class PredictiveAnalysisTestCase(unittest.TestCase):
    def test_linear_reg(self):
        actual_root_me, actual_r2e, actual_predicted_value, actual_output_df = pa.linear_reg(test_stock_data, 10,
                                                                                             'Amazon.com Inc')
        self.assertEqual(actual_root_me, 10.151)
        self.assertEqual(actual_r2e, 0.0978)
        self.assertEqual(actual_predicted_value, 3493.72)


if __name__ == '__main__':
    unittest.main()
