import unittest

from resonatoranalysis.analysis import *
from resonatoranalysis.file_handler import datapicker, gethdf5info


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.fname = "unit_testing/Test sets/2023-08-29 22-14-04 VNA Power.hdf5"
        self.data_dict, self.files = datapicker(self.fname)
        self.vna_info, self.temps = gethdf5info(self.fname)
        self.powers = (
            self.vna_info["VNA Power"] - self.vna_info["Variable Attenuator"] - 80
        )
        self.fit = fit_resonator_test([self.data_dict], [self.fname], list(self.powers))

    def test_fit_function(self):
        pass
