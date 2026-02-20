"""
OORAGAN
=======

An object oriented library to analyse and visualize resonator measurement data.

Provides:
  1. A ``Dataset`` object to fetch data and measurement info in data files
  2. A ``ResonatorFitter`` object to fit resonator measurement data
  3. A ``grapher`` function to create a Grapher object to visualize data or fit results
  4. A ``ResonatorAttribution`` class to help attributing measured resonator to physical resonator on sample
  5. A ``PPMSAnalysis`` class to analyse data from the PPMS and get critical temperature/magnetic field
"""

from .util import (
    strtime,
    convert_magphase_to_complex,
    convert_complex_to_magphase,
    load_graph_data,
)

from .parameters import Parameter, NullParameter
from .file_loading import Dataset, File
from .fitting import Fitter, FitResult
from .resonator_attribution import Resonator, ResonatorAttribution
from .ppms_analysis import PPMSAnalysis
from .plotting import triptych, quality_factors, losses, magnetic_field

__all__ = [
    "strtime",
    "convert_complex_to_magphase",
    "convert_magphase_to_complex",
    "load_graph_data",
    "Parameter",
    "NullParameter",
    "Dataset",
    "File",
    "Fitter",
    "FitResult",
    "Resonator",
    "ResonatorAttribution",
    "PPMSAnalysis",
    "triptych",
    "quality_factors",
    "losses",
    "magnetic_field",
]
