from numpy.typing import NDArray
from typing import Protocol, runtime_checkable


@runtime_checkable
class _FitResult(Protocol):
    """
    Type for fit results. Only for typing purposes.
    """

    @property
    def photon_nbr(self) -> NDArray: ...

    @property
    def Q_c(self) -> NDArray: ...

    @property
    def Q_c_error(self) -> NDArray: ...

    @property
    def Q_i(self) -> NDArray: ...

    @property
    def Q_i_error(self) -> NDArray: ...

    @property
    def Q_t(self) -> NDArray: ...

    @property
    def Q_t_error(self) -> NDArray: ...

    @property
    def f_r(self) -> NDArray: ...

    @property
    def f_r_error(self) -> NDArray: ...

    @property
    def omega_r(self) -> NDArray: ...

    @property
    def omega_r_error(self) -> NDArray: ...

    @property
    def internal_loss(self) -> NDArray: ...

    @property
    def internal_loss_error(self) -> NDArray: ...

    @property
    def coupling_loss(self) -> NDArray: ...

    @property
    def coupling_loss_error(self) -> NDArray: ...
