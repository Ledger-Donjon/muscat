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

def compute_cpa(
    traces: Trace,
    plaintexts: npt.NDArray[np.uint64],
    guess_range: int,
    target_byte: int,
    leakage_func: Callable[[int, int], int],
    batch_size: int,
) -> Cpa:
    """Compute the [`Cpa`] of the given traces."""

def compute_cpa_normal(
    traces: Trace,
    plaintexts: npt.NDArray[np.uint64],
    guess_range: int,
    target_byte: int,
    leakage_func: Callable[[int, int], int],
    batch_size: int,
) -> Cpa:
    """Compute the [`Cpa`] of the given traces."""

class Cpa:
    """Result of the CPA[^1] on some traces.

    [^1]: <https://www.iacr.org/archive/ches2004/31560016/31560016.pdf>
    """

    def rank(self) -> npt.NDArray[np.uint64]:
        """Rank guesses."""

    def corr(self) -> npt.NDArray[np.float32]:
        """Return the Pearson correlation coefficients."""

    def best_guess(self) -> int:
        """Return the guess with the highest Pearson correlation coefficient."""

    def max_corr(self) -> npt.NDArray[np.float32]:
        """Return the maximum Pearson correlation coefficient for each guess."""

class CpaProcessor:
    def __init__(self, num_samples: int, guess_range: int, dtype: np.dtype):
        """Create a new CPA processor."""

    def batch_update(
        self,
        trace_batch: Trace,
        plaintext_batch: npt.NDArray[np.uint64],
        leakage_model: Callable[[int, int], int],
    ):
        """Update the processor with a batch of traces and plaintexts."""

    def finalize(self, leakage_model: Callable[[int, int], int]) -> Cpa:
        """Finalize the computation and return the CPA result."""

class CpaNormalProcessor:
    def __init__(self, num_samples: int, batch_size: int, guess_range: int, dtype: np.dtype):
        """Create a new CPA processor."""

    def batch_update(
        self,
        trace_batch: Trace,
        plaintext_batch: npt.NDArray[np.uint64],
        leakage_model: Callable[[npt.NDArray[np.uint64], int], int],
    ):
        """Update the processor with a batch of traces and plaintexts."""

    def finalize(self) -> Cpa:
        """Finalize the computation and return the CPA result."""

def compute_dpa(
    traces: Trace,
    plaintexts: npt.NDArray[np.uint64],
    guess_range: int,
    selection_function: Callable[[int, int], bool],
    batch_size: int,
) -> Dpa:
    """Compute the [`Dpa`] of the given traces."""

class Dpa:
    """Result of the DPA[^1] on some traces.

    [^1]: <https://paulkocher.com/doc/DifferentialPowerAnalysis.pdf>
    """

    def rank(self) -> npt.NDArray[np.uint64]:
        """Return the rank of guesses"""

    def differential_curves(self) -> npt.NDArray[np.float32]:
        """Return the differential curves"""

    def best_guess(self) -> int:
        """Return the guess with the highest differential peak."""

    def max_differential_curves(self) -> npt.NDArray[np.float32]:
        """Return the maximum differential peak for each guess."""

class DpaProcessor:
    def __init__(self, num_samples: int, guess_range: int, dtype: np.dtype):
        """Create a new DPA processor."""

    def batch_update(
        self,
        trace_batch: Trace,
        plaintext_batch: npt.NDArray[np.uint64],
        selection_function: Callable[[int, int], bool],
    ):
        """Update the processor with a batch of traces and plaintexts."""

    def finalize(self) -> Dpa:
        """Finalize the computation and return the DPA result."""
