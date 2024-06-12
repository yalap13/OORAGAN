import unittest

from numpy import array

from resonatoranalysis.util import *
from resonatoranalysis.file_handler import datapicker, gethdf5info


class TestUtil(unittest.TestCase):
    def setUp(self):
        self.fname = "unit_testing/Test sets/2023-08-29 22-14-04 VNA Power.hdf5"
        self.data_dict, self.files = datapicker(self.fname)
        self.vna_info, self.temps = gethdf5info(self.fname)

    def test_calculate_power(self):
        calculated_power = calculate_power(
            -80, {self.fname: {"vna_info": self.vna_info, "temps": self.temps}}
        )
        self.assertEqual(
            list(calculated_power[self.fname]),
            [-100.0, -95.0, -90.0, -85.0, -80.0, -75.0, -70.0],
        )


if __name__ == "__main__":
    unittest.main()
