from ...simulator.display.calcul_tools import *
from ...simulator.display.draw import *


def φ0(p1, p2):
    """Compute the reference vector field at one point.

    Args:
        p1 (float): First coordinate
        p2 (float): Second coordinate

    Returns:
        tuple: Field components along p1 and p2
    """
    return -(p1**3 + p2**2 * p1 - p1 + p2), -(p2**3 + p1**2 * p2 - p1 - p2)


def φcw(p1, p2, c, D, k, r):
    """Clockwise vector field"""
    Z = D @ array([[1, 0], [0, -1]])
    Z_1 = inv(Z)
    z1 = Z_1[0, 0] * (p1 - c[0, 0]) + Z_1[0, 1] * (p2 - c[1, 0])
    z2 = Z_1[1, 0] * (p1 - c[0, 0]) + Z_1[1, 1] * (p2 - c[1, 0])
    w1, w2 = φ0(z1, z2)
    v1 = Z[0, 0] * w1 + Z[0, 1] * w2
    v2 = Z[1, 0] * w1 + Z[1, 1] * w2
    return v1, v2


def φccw(p1, p2, c, D, k, r):
    """Counterclockwise vector field"""
    D_1 = inv(D)
    z1 = D_1[0, 0] * (p1 - c[0, 0]) + D_1[0, 1] * (p2 - c[1, 0])
    z2 = D_1[1, 0] * (p1 - c[0, 0]) + D_1[1, 1] * (p2 - c[1, 0])
    w1, w2 = φ0(z1, z2)
    v1 = D[0, 0] * w1 + D[0, 1] * w2
    v2 = D[1, 0] * w1 + D[1, 1] * w2
    return v1, v2


def φrep(p1, p2, c, D, k, r):
    """Vector field repulsing to the center c of a circle"""
    φ1 = k * (p1 - c[0, 0])
    φ2 = k * (p2 - c[1, 0])
    return φ1, φ2
