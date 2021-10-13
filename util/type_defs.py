import numpy as np
import numpy.typing as npt
from typing import *


# Unfortunately, there is no real support for typing shapes of Numpy arrays.
# So, the closest I can get is giving different types descriptive names, but
# static analysis tools won't be able to verify the shapes

Int1D = npt.NDArray[np.uint8]       # 1D Int Array
Int2D_1C = npt.NDArray[np.uint8]    # 2D Int Array, 1 Channel. (h, w)
Int2D_3C = npt.NDArray[np.uint8]    # 2D Int Array, 3 Channels. (h, w, 3)
IntArray = npt.NDArray[np.uint8]    # Int Array, Unspecified size

Float1D = npt.NDArray[np.float]     # 1D Float Array
Float2D_1C = npt.NDArray[np.float]  # 2D Float Array, 1 Channel. (h, w)
Float2D_3C = npt.NDArray[np.float]  # 2D Float Array, 3 Channel. (h, w, 3)
FloatArray = npt.NDArray[np.float]  # Float Array, Unspecified size

Color = Int1D

Contour = np.ndarray
Histogram = List[Float1D]  # 3 Float1D Arrays, 1 for each channel BGR
