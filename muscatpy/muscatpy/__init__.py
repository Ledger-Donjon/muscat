from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import distinguishers, leakage_detection, leakage_model, processors
else:
    from .muscatpy import *

