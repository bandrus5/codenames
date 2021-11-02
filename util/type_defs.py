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

# 1D array of size 2 [X Y]
Point = Int1D

# 1D array of size 3 with color components [B G R]
Color = Int1D

# (N, 1, 2) Int array where N is the number of points in a contour
# Each pair represents the (X,Y) coordinate of a point on the contour
Contour = IntArray
# 3 Float1D Arrays, 1 for each channel BGR
# The length of each Float1D Array is the number of bins in the histogram (usually 256)
Histogram = List[Float1D]
