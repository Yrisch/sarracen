import math

import numpy as np
from numba import njit, prange


class BaseKernel:
    """A generic kernel used for data interpolation."""

    def __init__(self):
        self._ckernel_func_cache = None
        self._column_cache = None

    @staticmethod
    def get_radius() -> float:
        """Get the smoothing radius of this kernel."""
        return 1

    @staticmethod
    def w(q: float, dim: int) -> float:
        """Get the normalized weight of this kernel.

        Parameters
        ----------
        q : float
            The value to evaluate this kernel at.
        dim : {1, 2, 3}
            The number of dimensions to normalize the kernel value for.

        Returns
        -------
        float
            The normalized kernel weight at `q`.
        """

        return 1

    def get_column_kernel(self, samples: int = 1000) -> np.ndarray:
        """Integrate a given 3D kernel over the z-axis.

        Parameters
        ----------
        samples: int
            Number of sample points to calculate when approximating the kernel.

        Returns
        -------
            A ndarray of length (samples), containing the kernel approximation.

        Examples
        --------
        Use np.linspace and np.interp to use this column kernel approximation:
            np.interp(q,
                      np.linspace(0, kernel.get_radius(), samples),
                      column_kernel)
        """
        if samples == 1000 and self._column_cache is not None:
            return self._column_cache

        c_kernel = BaseKernel._int_func(self.get_radius(), samples, self.w)

        if samples == 1000:
            self._column_cache = c_kernel

        return c_kernel

    def get_column_kernel_func(self, samples):
        """Generate a numba-accelerated column kernel function.

        Creates a numba-accelerated function for column kernel weights. This
        function can be utilized similarly to kernel.w().

        Parameters
        ----------
        samples: int
            Number of sample points to calculate when approximating the kernel.

        Returns
        -------
        A numba-accelerated weight function.
        """
        if self._ckernel_func_cache is not None and samples == 1000:
            return self._ckernel_func_cache
        column_kernel = self.get_column_kernel(samples)
        radius = self.get_radius()

        # @njit(fastmath=True)
        # def func(q, dim):
        #     # using np.linspace() would break compatibility with the GPU
        #     # backend, so the calculation here is performed manually.
        #     wab_index = q * (samples - 1) / radius
        #     index = min(max(0, int(math.floor(wab_index))), samples - 1)
        #     index1 = min(max(0, int(math.ceil(wab_index))), samples - 1)
        #     t = wab_index - index
        #     return column_kernel[index] * (1 - t) + column_kernel[index1] * t

        return column_kernel

    # Internal function for performing the integral in _get_column_kernel()
    @staticmethod
    @njit(fastmath=False, parallel=False)
    def _int_func(radius, samples, wfunc):
        result = np.zeros(samples + 1)
        r2 = radius * radius

        for i in range(samples):
            q_xy2 = i * (r2 / samples)
            bounds = np.sqrt(r2 - q_xy2)
            dz = bounds / 99.0
            coldens = 0.0
            for j in range(100):
                q_z = j * dz
                q = np.sqrt(q_xy2 + q_z * q_z)
                y = wfunc(q, 3)
                if j == 0 or j == 99:
                    coldens += 0.5 * y * dz
                else:
                    coldens += y * dz
            result[i] = 2 * coldens

        result[samples] = 0.0

        return result
