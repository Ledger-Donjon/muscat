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

classes = plaintexts[:, 0] & 0b1 == 0b1

# Compute Welch's t-test using the `compute_ttest` helper
ttest = muscatpy.leakage_detection.compute_ttest(
    traces,
    classes,
    200,
)

plt.plot(ttest)
plt.title("T-test")
plt.show()
