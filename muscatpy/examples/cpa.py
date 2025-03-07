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

cpa = muscatpy.distinguishers.compute_cpa(
    traces,
    plaintexts,
    256,
    0,
    lambda guess, plaintext: hamming_weight(
        muscatpy.leakage_model.aes.sbox(guess ^ plaintext)
    ),
    200,
)

best_guess = cpa.best_guess()
print("Best subkey guess:", best_guess)

corr = cpa.corr()[best_guess, :]
plt.plot(corr)
plt.title("Pearson correlation coefficient")
plt.show()
