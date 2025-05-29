class Parameter:
    def __init__(self, range, name):
        self.range = range
        self.name = name


class VNABandwidth(Parameter):
    def __init__(self, range):
        super(VNABandwidth, self).__init__(range, "vna_bandwidth")


class VNAPower(Parameter):
    def __init__(self, range):
        super(VNAPower, self).__init__(range, "vna_power")


class VNAAverage(Parameter):
    def __init__(self, range):
        super(VNAAverage, self).__init__(range, "vna_average")


class MagneticField(Parameter):
    def __init__(self, range):
        super(MagneticField, self).__init__(range, "mag_field")
