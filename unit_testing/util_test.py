import unittest

from resonatoranalysis.util import calculate_power
from resonatoranalysis.file_handler import datapicker


class TestUtil(unittest.TestCase):
    def setUp(self):
        self.info_dict = {
            "C:/test/file/": {"vna_info": {"VNA Power": -30, "Variable Attenuator": 30}}
        }
        self.att_cryo = -80
        self.data_dict = {}

    def test_power_is_calculated(self):
        calculated_power = calculate_power(self.att_cryo, self.info_dict)
        self.assertDictEqual(calculated_power, {"C:/test/file/": -140})

    def test_freq_info(self):
        pass


if __name__ == "__main__":
    unittest.main()
