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

assert traces.shape[0] == plaintexts.shape[0]

# We convert from np.float64 to np.float32 as f64 is not supported
traces = traces.astype(np.float32)
plaintexts = plaintexts.astype(np.uint64)

# Compute the CPA using the `compute_cpa` helper
cpa = muscatpy.distinguishers.compute_cpa(
    traces,
    plaintexts,
    256,
    0,
    lambda plaintext, guess: hamming_weight(
        muscatpy.leakage_model.aes.sbox(plaintext ^ guess)
    ),
    200,
)

# Or using the lower level `CpaProcessor` class
leakage_model = lambda plaintext, guess: hamming_weight(
    muscatpy.leakage_model.aes.sbox(plaintext ^ guess)
)

cpa_processor = muscatpy.distinguishers.CpaProcessor(traces.shape[1], 256, traces.dtype)
for i in range(0, traces.shape[0], 200):
    cpa_processor.batch_update(
        traces[i : i + 200], plaintexts[i : i + 200, 0], leakage_model
    )
cpa = cpa_processor.finalize(leakage_model)

best_guess = cpa.best_guess()
print("Best subkey guess:", best_guess)

corr = cpa.corr()[best_guess, :]
plt.plot(corr)
plt.title("Pearson correlation coefficient")
plt.show()
