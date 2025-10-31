import functools
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import muscatpy


TRACES_DIR = Path(os.environ["TRACES_DIR"])

traces = np.load(TRACES_DIR / "traces.npy")
plaintexts = np.load(TRACES_DIR / "plaintexts.npy")

assert traces.shape[0] == plaintexts.shape[0]

# We convert from np.float64 to np.float32 as f64 is not supported
traces = traces.astype(np.float32)
plaintexts = plaintexts.astype(np.uint64)

# Compute the SNR using the `compute_snr` helper
snr = muscatpy.leakage_detection.compute_snr(
    traces,
    256,
    lambda i: plaintexts[i, 0],
    200,
)

plt.plot(snr)
plt.title("SNR")
plt.show()
