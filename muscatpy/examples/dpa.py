import functools
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import muscatpy


@functools.cache
def hamming_weight(x: int) -> int:
    weight = 0
    while x != 0:
        if x & 1 == 1:
            weight += 1
        x = x >> 1
    return weight


TRACES_DIR = Path(os.environ["TRACES_DIR"])

traces = np.load(TRACES_DIR / "traces.npy")
plaintexts = np.load(TRACES_DIR / "plaintexts.npy")

# We convert from np.float64 to np.float32 as f64 is not supported
traces = traces.astype(np.float32)
plaintexts = plaintexts.astype(np.uint64)

# Compute the DPA using the `compute_dpa` helper
dpa = muscatpy.distinguishers.compute_dpa(
    traces,
    plaintexts[:, 0],
    256,
    lambda plaintext, guess: (muscatpy.leakage_model.aes.sbox(plaintext ^ guess) & 1)
    == 1,
    200,
)

# Or using the lower level `DpaProcessor` class
selection_function = lambda plaintext, guess: (
    muscatpy.leakage_model.aes.sbox(plaintext ^ guess) & 1
) == 1
dpa_processor = muscatpy.distinguishers.DpaProcessor(traces.shape[1], 256, traces.dtype)
for i in range(0, traces.shape[0], 200):
    dpa_processor.batch_update(
        traces[i : i + 200], plaintexts[i : i + 200, 0], selection_function
    )
dpa = dpa_processor.finalize()

best_guess = dpa.best_guess()
print("Best subkey guess:", best_guess)

differential_curve = dpa.differential_curves()[best_guess, :]
plt.plot(differential_curve)
plt.title("Differential curve")
plt.show()
