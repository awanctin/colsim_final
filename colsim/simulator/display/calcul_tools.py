import matplotlib.pyplot as plt
from numpy import (
    mean,
    pi,
    cos,
    sin,
    sinc,
    sqrt,
    tan,
    arctan,
    arctan2,
    tanh,
    arcsin,
    arccos,
    exp,
    dot,
    array,
    log,
    inf,
    eye,
    zeros,
    ones,
    inf,
    arange,
    reshape,
    vstack,
    hstack,
    diag,
    median,
    sign,
    meshgrid,
    cross,
    linspace,
    append,
    rint,
)

from matplotlib.pyplot import *
from matplotlib.cbook import flatten
from numpy.random import randn, rand
from numpy.linalg import inv, det, eig, qr
from scipy.linalg import sqrtm, expm, logm, norm, block_diag

from scipy.signal import place_poles
from mpl_toolkits.mplot3d import Axes3D
from math import factorial
from matplotlib.patches import Ellipse, Rectangle, Circle, Wedge, Polygon, Arc
from matplotlib.collections import PatchCollection


def add1(M):
    """Append a row of ones to make homogeneous coordinates.

    Args:
        M (array): 2 x n matrix of points

    Returns:
        ndarray: 3 x n matrix
    """
    M = array(M)
    return vstack((M, ones(M.shape[1])))


def plot2D(ax, M, col="black", w=1):
    """Plot the first two rows of a matrix as a line.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on
        M (ndarray): Points, one per column
        col (str): Line colour
        w (float): Line width
    """
    ax.plot(M[0, :], M[1, :], col, linewidth=w)


def tran2H(x, y):
    """Build a 2D homogeneous translation matrix.

    Args:
        x (float): Shift along x
        y (float): Shift along y

    Returns:
        ndarray: 3 x 3 matrix
    """
    return array([[1, 0, x], [0, 1, y], [0, 0, 1]])


def rot2H(a):
    """Build a 2D homogeneous rotation matrix.

    Args:
        a (float): Rotation angle, rad

    Returns:
        ndarray: 3 x 3 matrix
    """
    return array([[cos(a), -sin(a), 0], [sin(a), cos(a), 0], [0, 0, 1]])


def arrow2H(L):
    """Build an arrow shape in homogeneous coordinates.

    Args:
        L (float): Arrow length

    Returns:
        ndarray: 3 x 5 matrix
    """
    e = 0.2
    return add1(L * array([[0, 1, 1 - e, 1, 1 - e], [0, 0, -e, 0, e]]))


def sawtooth(x):
    """Wrap an angle into [-pi, pi).

    Args:
        x (float): Angle, rad

    Returns:
        float: Wrapped angle, rad
    """
    return (x + pi) % (2 * pi) - pi  # or equivalently   2*arctan(tan(x/2))


def geo_scalar_prod(u, v, θu, θv):
    """Compute the dot product of two vectors given in polar form.

    Args:
        u (float): Norm of the first vector
        v (float): Norm of the second vector
        θu (float): Angle of the first vector, rad
        θv (float): Angle of the second vector, rad

    Returns:
        float: Dot product
    """
    return u * v * cos(θu - θv)


def dist(a, b):
    """Compute the distance between two positions.

    Args:
        a (ndarray): First state, x and y first
        b (ndarray): Second state, x and y first

    Returns:
        float: Euclidean distance
    """
    xa, ya = a[0:2].flatten()
    xb, yb = b[0:2].flatten()
    return sqrt((xb - xa) ** 2 + (yb - ya) ** 2)
