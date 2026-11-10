from .calcul_tools import *

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
    size,
    arange,
    reshape,
    vstack,
    hstack,
    diag,
    median,
    sign,
    sum,
    meshgrid,
    cross,
    linspace,
    append,
    round,
    trace,
    rint,
)
from matplotlib.pyplot import *
from matplotlib.cbook import flatten
from numpy.random import randn, rand
from numpy.linalg import inv, det, norm, eig, qr
from scipy.linalg import sqrtm, expm, logm, norm, block_diag

from scipy.signal import place_poles
from mpl_toolkits.mplot3d import Axes3D
from math import factorial
from matplotlib.patches import Ellipse, Rectangle, Circle, Wedge, Polygon, Arc
from matplotlib.collections import PatchCollection


def init_figure(xmin, xmax, ymin, ymax, width=10, height=10):
    """Create the simulation figure with fixed axis limits.

    Args:
        xmin (float): Lower x limit
        xmax (float): Upper x limit
        ymin (float): Lower y limit
        ymax (float): Upper y limit
        width (float): Figure width, unused
        height (float): Figure height, unused

    Returns:
        tuple: Figure and axes
    """
    fig, ax = plt.subplots()
    plt.suptitle("Simulation", size="x-large")

    ax.xmin = xmin
    ax.xmax = xmax
    ax.ymin = ymin
    ax.ymax = ymax
    clear(ax)
    return fig, ax


def init_table2(data, legend, xmin, xmax, ymin, ymax, text_size=11, length=4, width=2):
    """Create a new figure holding the rules table and its legend.

    Args:
        data (list): Table rows
        legend (list): Legend rows
        xmin (float): Unused
        xmax (float): Unused
        ymin (float): Unused
        ymax (float): Unused
        text_size (int): Font size
        length (float): Cell length scale
        width (float): Cell width scale

    Returns:
        tuple: Figure, axes and table
    """
    fig, ax = plt.subplots()
    plt.suptitle("Active rules of the sea", size="x-large")
    table = plt.table(cellText=data, loc="center")
    legend_table = plt.table(cellText=legend, loc="bottom")

    table.auto_set_font_size(False)
    table.set_fontsize(text_size)

    cell_colors = []
    for row in range(len(data)):
        current_row = []
        for col in range(len(data[row])):
            if (row, col) == (0, 0):
                current_row.append("lightgray")
            else:
                current_row.append("white")
        cell_colors.append(current_row)

    for row in range(len(data)):
        for col in range(len(data[row])):
            table[row, col].set_facecolor(cell_colors[row][col])

    ax.xmin = xmin
    ax.xmax = xmax
    ax.ymin = ymin
    ax.ymax = ymax
    clear(ax)
    ax.axis("off")
    fig.tight_layout()
    return fig, ax, table


def init_table(rules, legend, fig, ax, text_size=11, length=4, width=2):
    """Draw the rules table and its legend on existing axes.

    Args:
        rules (list): Table rows
        legend (list): Legend rows
        fig (matplotlib.figure.Figure): Figure to draw on
        ax (matplotlib.axes.Axes): Axes to draw on
        text_size (int): Font size
        length (float): Cell length scale
        width (float): Cell width scale

    Returns:
        matplotlib.table.Table: The rules table
    """
    plt.suptitle("Active rules of the sea", size="x-large")
    table = plt.table(cellText=rules, loc="center")
    legend_table = plt.table(cellText=legend, loc="bottom")
    table.auto_set_font_size(False)
    table.set_fontsize(text_size)
    cell_colors = []
    for row in range(len(rules)):
        current_row = []
        for col in range(len(rules[row])):
            if (row, col) == (row, 0):
                current_row.append("lightgray")
            else:
                current_row.append("white")
        cell_colors.append(current_row)

    for row in range(len(rules)):
        for col in range(len(rules[row])):
            table[row, col].set_facecolor(cell_colors[row][col])

    ax.axis("off")
    fig.tight_layout()
    return table


def clear(ax):
    """Clear the axes and reset their limits.

    Args:
        ax (matplotlib.axes.Axes): Axes to clear
    """
    pause(0.001)
    ax.cla()
    ax.set_xlim(ax.xmin, ax.xmax)
    ax.set_ylim(ax.ymin, ax.ymax)


def draw_arrow(ax, x, y, θ, L, col="darkblue", w=1):
    """Draw an arrow from a point along a heading.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on
        x (float): Start x
        y (float): Start y
        θ (float): Heading, rad
        L (float): Arrow length
        col (str): Colour
        w (float): Line width
    """
    plot2D(ax, tran2H(x, y) @ rot2H(θ) @ arrow2H(L), col, w)


def draw_boat_and_vector(ax, x, col="darkblue", r=0.1, w=2):
    """Draw a small boat shape and its speed arrow.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on
        x (ndarray): State vector (x, y, v, heading)
        col (str): Colour
        r (float): Shape scale
        w (float): Line width
    """
    mx, my, v, θ = list(x[0:4, 0])
    M = r * array([[-1, 5, 7, 7, 5, -1, -1, -1], [-2, -2, -1, 1, 2, 2, -2, -2]])
    M = add1(M)
    draw_arrow(ax, mx, my, θ, norm(v), "red")
    plot2D(ax, tran2H(mx, my) @ rot2H(θ) @ M, col, w)


def draw_field_around_c(ax, f, xmin, xmax, ymin, ymax, a, c):
    """Draw a normalised vector field around a centre.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on
        f (callable): Field function f(X1, X2, c)
        xmin (float): Lower x limit
        xmax (float): Upper x limit
        ymin (float): Lower y limit
        ymax (float): Upper y limit
        a (float): Grid step
        c (ndarray): Field centre
    """
    Mx = arange(xmin, xmax, a)
    My = arange(ymin, ymax, a)
    X1, X2 = meshgrid(Mx, My)
    VX, VY = f(X1, X2, c)
    R = sqrt(VX**2 + VY**2)
    ax.quiver(Mx, My, VX / R, VY / R)


def draw_field_around_c_new(ax, f, xmin, xmax, ymin, ymax, a, c, D, k, r):
    """Draw a normalised vector field with extra field parameters.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on
        f (callable): Field function f(X1, X2, c, D, k, r)
        xmin (float): Lower x limit
        xmax (float): Upper x limit
        ymin (float): Lower y limit
        ymax (float): Upper y limit
        a (float): Grid step
        c (ndarray): Field centre
        D (float): Field parameter passed to f
        k (float): Field gain passed to f
        r (float): Field radius passed to f
    """
    Mx = arange(xmin, xmax, a)
    My = arange(ymin, ymax, a)
    X1, X2 = meshgrid(Mx, My)
    VX, VY = f(X1, X2, c, D, k, r)
    R = sqrt(VX**2 + VY**2)
    ax.quiver(Mx, My, VX / R, VY / R)


def draw_circle(ax, center_x, center_y, radius, color, legend=None, alpha=1):
    """Draw an empty circle.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on
        center_x (float): Centre x
        center_y (float): Centre y
        radius (float): Radius
        color (str): Colour
        legend (str): Unused
        alpha (float): Transparency
    """
    circle = plt.Circle(
        (center_x, center_y), radius, fill=False, color=color, alpha=alpha
    )
    ax.add_artist(circle)


def draw_disk(ax, c, r, col, alph=0.7, w=1):
    """Draw a filled disc.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on
        c (tuple): Centre (x, y)
        r (float): Radius
        col (str): Fill colour
        alph (float): Transparency
        w (float): Edge width
    """
    e = Ellipse(xy=c, width=2 * r, height=2 * r, angle=0, linewidth=w)
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(alph)
    e.set_facecolor(col)


def draw_collision(
    ax, center_x1, center_y1, center_x2, center_y2, radius=1, color="red"
):
    """Mark a collision with two dashed circles and two crosses.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on
        center_x1 (float): First vessel x
        center_y1 (float): First vessel y
        center_x2 (float): Second vessel x
        center_y2 (float): Second vessel y
        radius (float): Circle radius
        color (str): Colour
    """
    circle1 = plt.Circle(
        (center_x1, center_y1), radius, fill=False, color=color, linestyle="--"
    )
    circle2 = plt.Circle(
        (center_x2, center_y2), radius, fill=False, color=color, linestyle="--"
    )
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.plot(center_x1, center_y1, "rx")
    ax.plot(center_x2, center_y2, "rx")
