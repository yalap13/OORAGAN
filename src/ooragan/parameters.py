from typing import Self


class Parameter:
    """
    Analogous to the parameters used by MeaVis. The name should be the same as
    the one defined for the MeaVis parameters.
    """

    def __init__(self, range, name):
        self.range = range
        self._meavis_name = name

    def _get_existing_parameters(self) -> list[str]:
        subclasses = Self.__subclasses__()

        def get_parameter_key(t: type) -> str:
            return getattr(t.__name__, "_key")

        parameters = map(get_parameter_key, subclasses)


class VNABandwidth(Parameter):
    def __init__(self, range):
        super(VNABandwidth, self).__init__(range, "VNA Bandwidth")
        self._key = "vna_bandwidth"


class VNAPower(Parameter):
    def __init__(self, range):
        super(VNAPower, self).__init__(range, "VNA Power")
        self._key = "vna_power"


class VNAAverage(Parameter):
    def __init__(self, range):
        super(VNAAverage, self).__init__(range, "VNA Average")
        self._key = "vna_average"


class MagneticField(Parameter):
    def __init__(self, range):
        super(MagneticField, self).__init__(range, "Magnet")
        self._key = "magnetic_field"


class Index(Parameter):
    def __init__(self, range):
        super(Index, self).__init__(range, "Index")
        self._key = "index"


class DigitalAttenuation(Parameter):
    def __init__(self, range):
        super(DigitalAttenuation, self).__init__(range, "Digital Attenuator")
        self._key = "dig_attenuation"
