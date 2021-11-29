import unittest
import pandas as pd
import stocks_data as sd
from datetime import datetime

test_stock_data = pd.read_csv('amazon_test_data.csv', index_col=0)
test_stock_data.index = pd.to_datetime(test_stock_data.index).date


class StockDataTestCase(unittest.TestCase):

    def test_get_stock_data(self):
        actual_output = sd.get_stock_data('AMZN', datetime.strptime('28-08-2021', '%d-%m-%Y').date(),
                                          datetime.strptime('29-11-2021', '%d-%m-%Y').date())
        expected_output = test_stock_data
        pd.testing.assert_frame_equal(expected_output, actual_output, check_names=False)


if __name__ == '__main__':
    unittest.main()
