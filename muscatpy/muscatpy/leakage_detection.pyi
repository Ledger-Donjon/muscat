from collections.abc import Callable
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

def compute_snr(
    traces: Trace,
    classes: int,
    get_class: Callable[[int], int],
    batch_size: int,
) -> npt.NDArray[np.float32]:
    """Compute the SNR of the given traces."""

def compute_nicv(
    traces: Trace,
    classes: int,
    get_class: Callable[[int], int],
    batch_size: int,
) -> npt.NDArray[np.float32]:
    """Compute the NICV of the given traces."""

def compute_ttest(
    traces: Trace,
    trace_classes: npt.NDArray[np.bool_],
    batch_size: int,
) -> npt.NDArray[np.float32]:
    """Compute the Welch's T-test of the given traces."""
