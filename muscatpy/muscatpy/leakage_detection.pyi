from collections.abc import Callable

import numpy as np
import numpy.typing as npt

def compute_snr(
    leakages: npt.NDArray[np.int64],
    classes: int,
    get_class: Callable[[int], int],
    batch_size: int,
) -> npt.NDArray[np.float64]:
    """Compute the SNR of the given traces."""

def compute_ttest(
    traces: npt.NDArray[np.int64],
    trace_classes: npt.NDArray[np.bool_],
    batch_size: int,
) -> npt.NDArray[np.float64]:
    """Compute the Welch's T-test of the given traces."""
