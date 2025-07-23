from typing import Optional
from numpy.typing import ArrayLike
from numpy import empty


class Parameter:
    """
    Analogous to the parameters used by MeaVis. The name should be the same as
    the one defined for the MeaVis parameters.

    Parameters
    ----------
    range : ArrayLike
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
        range: ArrayLike,
        name: str,
        description: Optional[str] = None,
        unit: Optional[str] = None,
    ):
        self.range = range
        self.name = name
        self.description = description
        self.unit = unit


class NullParameter(Parameter):
    """Empty parameter place holder"""

    def __init__(self):
        super(NullParameter, self).__init__(empty(0), "null")
