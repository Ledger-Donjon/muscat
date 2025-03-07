from typing import Union

import numpy as np
import numpy.typing as npt

Trace = Union[
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint16],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint64],
    npt.NDArray[np.int8],
    npt.NDArray[np.int16],
    npt.NDArray[np.int32],
    npt.NDArray[np.int64],
    npt.NDArray[np.float32],
]

def compute_mean(
    traces: Trace,
) -> npt.NDArray[np.float32]:
    """Compute the mean of the given traces."""

def compute_var(
    traces: Trace,
) -> npt.NDArray[np.float32]:
    """Compute the variance of the given traces."""
