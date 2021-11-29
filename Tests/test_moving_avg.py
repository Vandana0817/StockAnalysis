import unittest
import pandas as pd
import numpy as np
import moving_averages as ma

test_stock_data = pd.read_csv('amazon_test_data.csv', index_col=0)
test_stock_data.index = pd.to_datetime(test_stock_data.index).date


class MovingAverageTestCase(unittest.TestCase):

    def test_calculate_wma(self):
        weight_list = list(reversed([(10 - n) * 10 for n in range(10)]))
        actual_value = ma.calculate_wma(weight_list)
        self.assertEqual(actual_value(5), 5.0)

    def test_compute_adj_close_data(self):
        actual_value = ma.compute_adj_close_data(test_stock_data)
        self.assertEqual(True, actual_value.equals(test_stock_data['Adj Close']))

    def test_compute_moving_averages(self):
        expected_output_data = {
            'Date': ['2021-08-30', '2021-08-31', '2021-09-01', '2021-09-02', '2021-09-03'],
            'Adj Close': [3421.570068, 3470.790039, 3479.000000, 3463.120117, 3478.050049],
            'SMA': [np.nan, np.nan, 3462.506055, 3480.050049, 3490.992041],
            'WMA': [np.nan, np.nan, np.nan, np.nan, 3469.525391],
            'EMA': [3421.570068, 3437.976725, 3451.651150, 3455.474139, 3462.999442],
            'MACD': [0.000000, -0.785276, -2.150777, -3.550115, -5.127411],
        }
        expected_output = pd.DataFrame(expected_output_data)
        expected_output.set_index('Date', inplace=True)
        expected_output.index = pd.to_datetime(expected_output.index).date
        actual_value = ma.compute_moving_averages(test_stock_data, 'Adj Close', 5)
        pd.testing.assert_frame_equal(expected_output.head(), actual_value.head(), check_names=False)


if __name__ == '__main__':
    unittest.main()
