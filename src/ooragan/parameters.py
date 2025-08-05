from typing import Optional
from numpy.typing import NDArray
from numpy import empty, float64


class Parameter:
    """
    Analogous to the parameters used by MeaVis. The name should be the same as
    the one defined for the MeaVis parameters.

    Parameters
    ----------
    range : NDArray of float64
        Range of values for this parameter.
    name : str
        Name of the parameter. For parameters existing in MeaVis, should be the
        same name.
    description : str, optional
        Description of the parameter.
    unit : str, optional
        Unit of the parameter.
    """

    def __init__(
        self,
        range: NDArray[float64],
        name: str,
        description: Optional[str] = None,
        unit: Optional[str] = None,
    ):
        self.range = range
        self.name = name
        self.description = description
        self.unit = unit

    def __repr__(self) -> str:
        return f"ooragan.Parameter({self.name}, {self.range.shape}, {self.description}, {self.unit})"


class NullParameter(Parameter):
    """Empty parameter place holder"""

    def __init__(self):
        super(NullParameter, self).__init__(empty(0), "null")
