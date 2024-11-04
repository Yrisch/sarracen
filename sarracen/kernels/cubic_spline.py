import numpy as np
from numba import njit

from ..kernels import BaseKernel


class CubicSplineKernel(BaseKernel):
    """An implementation of the Cubic Spline kernel"""

    @staticmethod
    def get_radius() -> float:
        return 2.0

    @staticmethod
    @njit(fastmath=True, cache=True)
    def w(q: float, ndim: int):
        norm = (
            2.0 / 3.0
            if (ndim == 1)
            else 10.0 / (7.0 * np.pi) if (ndim == 2) else 1.0 / np.pi
        )

        return norm * (
            (1.0 - 1.5 * q * q + 0.75 * q * q * q) * (0.0 <= q) * (q < 1.0)
            + 0.25 * (2.0 - q) * (2.0 - q) * (2.0 - q) * (1.0 <= q) * (q < 2.0)
        )
