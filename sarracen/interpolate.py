import numpy as np
from numba import prange, njit

from sarracen.kernels import BaseKernel


def interpolate_2d(data: 'SarracenDataFrame',
                   target: str,
                   x: str,
                   y: str,
                   kernel: BaseKernel,
                   pixwidthx: float,
                   pixwidthy: float,
                   xmin: float = 0,
                   ymin: float = 0,
                   pixcountx: int = 480,
                   pixcounty: int = 480):
    """
    Interpolates particle data in a SarracenDataFrame across two directional axes to a 2D
    grid of pixels.

    :param data: The particle data, in a SarracenDataFrame.
    :param x: The column label of the x-directional axis.
    :param y: The column label of the y-directional axis.
    :param target: The column label of the target smoothing data.
    :param kernel: The kernel to use for smoothing the target data.
    :param pixwidthx: The width that each pixel represents in particle data space.
    :param pixwidthy: The height that each pixel represents in particle data space.
    :param xmin: The starting x-coordinate (in particle data space).
    :param ymin: The starting y-coordinate (in particle data space).
    :param pixcountx: The number of pixels in the output image in the x-direction.
    :param pixcounty: The number of pixels in the output image in the y-direction.
    :return: The output image, in a 2-dimensional numpy array.
    """
    if pixwidthx <= 0:
        raise ValueError("pixwidthx must be greater than zero!")
    if pixwidthy <= 0:
        raise ValueError("pixwidthy must be greater than zero!")
    if pixcountx <= 0:
        raise ValueError("pixcountx must be greater than zero!")
    if pixcounty <= 0:
        raise ValueError("pixcounty must be greater than zero!")

    return _fast_2d(data[target].to_numpy(), data[x].to_numpy(), data[y].to_numpy(), kernel.weight, kernel.get_radius(),
                    data['m'].to_numpy(), data['rho'].to_numpy(), data['h'].to_numpy(), xmin, ymin, pixwidthx,
                    pixwidthy, pixcountx, pixcounty)


# Underlying numba-compiled code for 2D interpolation
@njit(parallel=True, fastmath=True)
def _fast_2d(target, x_data, y_data, wfunc, k_rad, mass, rho, h, xmin, ymin, pixwidthx, pixwidthy,
             pixcountx, pixcounty):
    image = np.zeros((pixcounty, pixcountx))

    term = (target * mass / (rho * h ** 2))

    # determine maximum and minimum pixels that each particle contributes to
    ipixmin = np.rint((x_data - k_rad * h - xmin) / pixwidthx) \
        .clip(a_min=0, a_max=pixcountx)
    jpixmin = np.rint((y_data - k_rad * h - ymin) / pixwidthy) \
        .clip(a_min=0, a_max=pixcounty)
    ipixmax = np.rint((x_data + k_rad * h - xmin) / pixwidthx) \
        .clip(a_min=0, a_max=pixcountx)
    jpixmax = np.rint((y_data + k_rad * h - ymin) / pixwidthy) \
        .clip(a_min=0, a_max=pixcounty)

    # iterate through the indexes of non-filtered particles
    for i in prange(len(term)):
        # precalculate differences in the x-direction (optimization)
        dx2i = ((xmin + (np.arange(int(ipixmin[i]), int(ipixmax[i])) + 0.5) * pixwidthx - x_data[i]) ** 2) \
               * (1 / (h[i] ** 2))

        # determine differences in the y-direction
        ypix = ymin + (np.arange(int(jpixmin[i]), int(jpixmax[i])) + 0.5) * pixwidthy
        dy = ypix - y_data[i]
        dy2 = dy * dy * (1 / (h[i] ** 2))

        # calculate contributions at pixels i, j due to particle at x, y
        q2 = dx2i + dy2.reshape(len(dy2), 1)
        wab = wfunc(np.sqrt(q2), 2)

        # add contributions to image
        image[int(jpixmin[i]):int(jpixmax[i]), int(ipixmin[i]):int(ipixmax[i])] += (wab * term[i])

    return image


def interpolate_2d_cross(data: 'SarracenDataFrame',
                         target: str,
                         x: str,
                         y: str,
                         kernel: BaseKernel,
                         x1: float = 0,
                         y1: float = 0,
                         x2: float = 1,
                         y2: float = 1,
                         pixcount: int = 500) -> np.ndarray:
    """
    Interpolates particle data in a SarracenDataFrame across two directional axes to a 1D
    cross-section line.

    :param data: The particle data, in a SarracenDataFrame.
    :param x: The column label of the x-directional axis.
    :param y: The column label of the y-directional axis.
    :param target: The column label of the target smoothing data.
    :param kernel: The kernel to use for smoothing the target data.
    :param x1: The starting x-coordinate of the cross-section line. (in particle data space)
    :param y1: The starting y-coordinate of the cross-section line. (in particle data space)
    :param x2: The ending x-coordinate of the cross-section line. (in particle data space)
    :param y2: The ending y-coordinate of the cross-section line. (in particle data space)
    :param pixcount: The number of pixels in the output over the entire cross-sectional line.
    :return: The interpolated output, in a 1-dimensional numpy array.
    """
    if np.isclose(y2, y1) and np.isclose(x2, x1):
        raise ValueError('Zero length cross section!')

    if pixcount <= 0:
        raise ValueError('pixcount must be greater than zero!')

    return _fast_2d_cross(data[target].to_numpy(), data[x].to_numpy(), data[y].to_numpy(), kernel.weight,
                          kernel.get_radius(), data['m'].to_numpy(), data['rho'].to_numpy(), data['h'].to_numpy(), x1,
                          y1, x2, y2, pixcount)


# Underlying numba-compiled code for 2D->1D cross-sections
@njit(parallel=True, fastmath=True)
def _fast_2d_cross(target, x_data, y_data, wfunc, k_rad, mass, rho, h, x1, y1, x2, y2, pixcount):
    # determine the slope of the cross-section line
    gradient = 0
    if not x2 - x1 == 0:
        gradient = (y2 - y1) / (x2 - x1)
    yint = y2 - gradient * x2

    # determine the fraction of the line that one pixel represents
    xlength = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    pixwidth = xlength / pixcount
    xpixwidth = (x2 - x1) / pixcount

    term = target * mass / (rho * h ** 2)

    # the intersections between the line and a particle's 'smoothing circle' are
    # found by solving a quadratic equation with the below values of a, b, and c.
    # if the determinant is negative, the particle does not contribute to the
    # cross-section, and can be removed.
    aa = 1 + gradient ** 2
    bb = 2 * gradient * (yint - y_data) - 2 * x_data
    cc = x_data ** 2 + y_data ** 2 - 2 * yint * y_data + yint ** 2 - (k_rad * h) ** 2
    det = bb ** 2 - 4 * aa * cc

    # create a filter for particles that do not contribute to the cross-section
    filter_det = det >= 0
    det = np.sqrt(det)
    cc = None

    output = np.zeros(pixcount)

    # the starting and ending x coordinates of the lines intersections with a particle's smoothing circle
    xstart = ((-bb[filter_det] - det[filter_det]) / (2 * aa)).clip(a_min=x1, a_max=x2)
    xend = ((-bb[filter_det] + det[filter_det]) / (2 * aa)).clip(a_min=x1, a_max=x2)
    bb, det = None, None

    # the start and end distances which lie within a particle's smoothing circle.
    rstart = np.sqrt((xstart - x1) ** 2 + ((gradient * xstart + yint) - y1) ** 2)
    rend = np.sqrt((xend - x1) ** 2 + (((gradient * xend + yint) - y1) ** 2))
    xstart, xend = None, None

    # the maximum and minimum pixels that each particle contributes to.
    ipixmin = np.rint(rstart / pixwidth).clip(a_min=0, a_max=pixcount)
    ipixmax = np.rint(rend / pixwidth).clip(a_min=0, a_max=pixcount)
    rstart, rend = None, None

    # iterate through the indices of all non-filtered particles
    for i in prange(len(x_data[filter_det])):
        # determine contributions to all affected pixels for this particle
        xpix = x1 + (np.arange(int(ipixmin[i]), int(ipixmax[i])) + 0.5) * xpixwidth
        ypix = gradient * xpix + yint
        dy = ypix - y_data[filter_det][i]
        dx = xpix - x_data[filter_det][i]

        q2 = (dx * dx + dy * dy) * (1 / (h[filter_det][i] * h[filter_det][i]))
        wab = wfunc(np.sqrt(q2), 2)

        # add contributions to output total, transformed by minimum/maximum pixels
        output[int(ipixmin[i]):int(ipixmax[i])] += (wab * term[filter_det][i])

    return output


def interpolate_3d(data: 'SarracenDataFrame',
                   target: str,
                   x: str,
                   y: str,
                   kernel: BaseKernel,
                   pixwidthx: float,
                   pixwidthy: float,
                   xmin: float = 0,
                   ymin: float = 0,
                   pixcountx: int = 480,
                   pixcounty: int = 480,
                   int_samples: int = 1000):
    """ Interpolate 3D particle data to a 2D grid of pixels.

    Interpolates three-dimensional particle data in a SarracenDataFrame. The data
    is interpolated to a 2D grid of pixels, by summing contributions in columns which
    span the z-axis.

    Parameters
    ----------
    data : SarracenDataFrame
        The particle data, in a SarracenDataFrame.
    x: str
        The column label of the x-directional axis.
    y: str
        The column label of the y-directional axis.
    target: str
        The column label of the target smoothing data.
    kernel: BaseKernel
        The kernel to use for smoothing the target data.
    pixwidthx: float
        The width that each pixel represents in particle data space.
    pixwidthy: float
        The height that each pixel represents in particle data space.
    xmin: float, optional
        The starting x-coordinate (in particle data space).
    ymin: float, optional
        The starting y-coordinate (in particle data space).
    pixcountx: int, optional
        The number of pixels in the output image in the x-direction.
    pixcounty: int, optional
        The number of pixels in the output image in the y-direction.
    int_samples: int, optional
        The number of sample points to take when approximating the 2D column kernel.

    Returns
    -------
    ndarray
        The interpolated output image, in a 2-dimensional numpy array.

    Raises
    -------
    ValueError
        If `pixwidthx`, `pixwidthy`, `pixcountx`, or `pixcounty` are less than or equal to zero.
    """
    if pixwidthx <= 0:
        raise ValueError("pixwidthx must be greater than zero!")
    if pixwidthy <= 0:
        raise ValueError("pixwidthy must be greater than zero!")
    if pixcountx <= 0:
        raise ValueError("pixcountx must be greater than zero!")
    if pixcounty <= 0:
        raise ValueError("pixcounty must be greater than zero!")

    return _fast_3d(data[target].to_numpy(), data[x].to_numpy(), data[y].to_numpy(),
                    kernel.get_column_kernel(int_samples), int_samples, kernel.get_radius(), data['m'].to_numpy(),
                    data['rho'].to_numpy(), data['h'].to_numpy(), xmin, ymin, pixwidthx, pixwidthy, pixcountx,
                    pixcounty)


# Underlying numba-compiled code for 3D column interpolation.
@njit(parallel=True, fastmath=True)
def _fast_3d(target, x_data, y_data, wfuncint, int_samples, k_rad, mass, rho, h, xmin, ymin,
             pixwidthx, pixwidthy, pixcountx, pixcounty):
    image = np.zeros((pixcounty, pixcountx))

    term = target * mass / (rho * h ** 2)

    # determine maximum and minimum pixels that each particle contributes to
    ipixmin = np.rint((x_data - k_rad * h - xmin) / pixwidthx) \
        .clip(a_min=0, a_max=pixcountx)
    jpixmin = np.rint((y_data - k_rad * h - ymin) / pixwidthy) \
        .clip(a_min=0, a_max=pixcounty)
    ipixmax = np.rint((x_data + k_rad * h - xmin) / pixwidthx) \
        .clip(a_min=0, a_max=pixcountx)
    jpixmax = np.rint((y_data + k_rad * h - ymin) / pixwidthy) \
        .clip(a_min=0, a_max=pixcounty)

    # iterate through the indexes of non-filtered particles
    for i in prange(len(term)):
        # precalculate differences in the x-direction (optimization)
        dx2i = ((xmin + (np.arange(int(ipixmin[i]), int(ipixmax[i])) + 0.5) * pixwidthx - x_data[i]) ** 2) \
               * (1 / (h[i] ** 2))

        # determine differences in the y-direction
        ypix = ymin + (np.arange(int(jpixmin[i]), int(jpixmax[i])) + 0.5) * pixwidthy
        dy = ypix - y_data[i]
        dy2 = dy * dy * (1 / (h[i] ** 2))

        # calculate contributions at pixels i, j due to particle at x, y
        q2 = dx2i + dy2.reshape(len(dy2), 1)
        wab = np.interp(np.sqrt(q2), np.linspace(0, k_rad, int_samples), wfuncint)

        # add contributions to image
        image[int(jpixmin[i]):int(jpixmax[i]), int(ipixmin[i]):int(ipixmax[i])] += (wab * term[i])

    return image


def interpolate_3d_cross(data: 'SarracenDataFrame',
                         target: str,
                         zslice: float,
                         x: str,
                         y: str,
                         z: str,
                         kernel: BaseKernel,
                         pixwidthx: float,
                         pixwidthy: float,
                         xmin: float = 0,
                         ymin: float = 0,
                         pixcountx: int = 480,
                         pixcounty: int = 480):
    """ Interpolate 3D particle data to a 2D grid, using a 3D cross-section.

    Interpolates particle data in a SarracenDataFrame across three directional axes to a 2D
    grid of pixels. A cross-section is taken of the 3D data at a specific value of z, and
    the contributions of particles near the plane are interpolated to a 2D grid.

    Parameters
    ----------
    data : SarracenDataFrame
        The particle data to interpolate over.
    x: str
        The column label of the x-directional axis.
    y: str
        The column label of the y-directional axis.
    z: str
        The column label of the z-directional axis.
    target: str
        The column label of the target smoothing data.
    kernel: BaseKernel
        The kernel to use for smoothing the target data.
    zslice: float
        The z-axis value to take the cross-section at.
    pixwidthx: float
        The width that each pixel represents in particle data space.
    pixwidthy: float
        The height that each pixel represents in particle data space.
    xmin: float, optional
        The starting x-coordinate (in particle data space).
    ymin: float, optional
        The starting y-coordinate (in particle data space).
    pixcountx: int, optional
        The number of pixels in the output image in the x-direction.
    pixcounty: int, optional
        The number of pixels in the output image in the y-direction.

    Returns
    -------
    ndarray
        The interpolated output image, in a 2-dimensional numpy array. Dimensions are
        structured in reverse order, where (x, y) -> [y][x]

    Raises
    -------
    ValueError
        If `pixwidthx`, `pixwidthy`, `pixcountx`, or `pixcounty` are less than or equal to zero.
    """
    if pixwidthx <= 0:
        raise ValueError("pixwidthx must be greater than zero!")
    if pixwidthy <= 0:
        raise ValueError("pixwidthy must be greater than zero!")
    if pixcountx <= 0:
        raise ValueError("pixcountx must be greater than zero!")
    if pixcounty <= 0:
        raise ValueError("pixcounty must be greater than zero!")

    return _fast_3d_cross(data[target].to_numpy(), data[x].to_numpy(), data[y].to_numpy(), data[z].to_numpy(),
                          kernel.weight, zslice, kernel.get_radius(), data['m'].to_numpy(), data['rho'].to_numpy(),
                          data['h'].to_numpy(), pixwidthx, pixwidthy, xmin, ymin, pixcountx, pixcounty)


# Underlying numba-compiled code for 3D->2D cross-sections
@njit(parallel=True, fastmath=True)
def _fast_3d_cross(target, x_data, y_data, z_data, wfunc, zslice, k_rad, mass, rho, h, pixwidthx, pixwidthy,
                   xmin, ymin, pixcountx, pixcounty):
    # Filter out particles that do not contribute to this cross-section slice
    term = target * mass / (rho * h ** 3)
    dz = zslice - z_data

    filter_distance = np.abs(dz) < k_rad * h

    ipixmin = np.rint((x_data[filter_distance] - k_rad * h[filter_distance] - xmin) / pixwidthx).clip(a_min=0,
                                                                                                      a_max=pixcountx)
    jpixmin = np.rint((y_data[filter_distance] - k_rad * h[filter_distance] - ymin) / pixwidthy).clip(a_min=0,
                                                                                                      a_max=pixcounty)
    ipixmax = np.rint((x_data[filter_distance] + k_rad * h[filter_distance] - xmin) / pixwidthx).clip(a_min=0,
                                                                                                      a_max=pixcountx)
    jpixmax = np.rint((y_data[filter_distance] + k_rad * h[filter_distance] - ymin) / pixwidthy).clip(a_min=0,
                                                                                                      a_max=pixcounty)

    image = np.zeros((pixcounty, pixcountx))

    for i in prange(len(x_data[filter_distance])):
        # precalculate differences in the x-direction
        dx2i = (((xmin + (np.arange(int(ipixmin[i]), int(ipixmax[i])) + 0.5)
                  * pixwidthx - x_data[filter_distance][i]) ** 2)
                * (1 / (h[filter_distance][i] ** 2))) + (
                       (dz[filter_distance][i] ** 2) * (1 / h[filter_distance][i] ** 2))

        ypix = ymin + (np.arange(int(jpixmin[i]), int(jpixmax[i])) + 0.5) * pixwidthy
        dy = ypix - y_data[filter_distance][i]
        dy2 = dy * dy * (1 / (h[filter_distance][i] ** 2))

        q2 = dx2i + dy2.reshape(len(dy2), 1)
        contribution = (term[filter_distance][i] * wfunc(np.sqrt(q2), 3))
        image[int(jpixmin[i]):int(jpixmax[i]), int(ipixmin[i]):int(ipixmax[i])] += contribution

    return image
