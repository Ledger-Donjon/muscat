from collections.abc import Callable

import numpy as np
import numpy.typing as npt

def compute_cpa(
    traces: npt.NDArray,
    plaintexts: npt.NDArray[np.uint64],
    guess_range: int,
    target_byte: int,
    leakage_func: Callable[[int, int], int],
    batch_size: int,
) -> Cpa:
    """Compute the [`Cpa`] of the given traces."""

def compute_cpa_normal(
    traces: npt.NDArray,
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

def compute_dpa(
    traces: npt.NDArray,
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
