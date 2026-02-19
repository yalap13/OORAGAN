from typing import Protocol, runtime_checkable


@runtime_checkable
class _FitResult(Protocol):
    """
    Type for fit results. Only for typing purposes.
    """
