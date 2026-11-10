"""Run Imazu case 2 with RRT* and save a static figure and a GIF.

Usage, from the repository root:
    python examples/imazu2.py

Writes out/imazu2.png and out/imazu2.gif.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from colsim.simulator.simulation_runner import SimulationRunner

IMAZU_CASE = 2
PLANNER = "RRT*"
NUM_STEPS = 160
MAP_HALF_EXTENT_M = 11_000.0
OUT_DIR = "out"
GIF_FPS = 15
FIGURE_SIZE_IN = (6, 6)
PNG_DPI = 150
TRACK_LABELS = ("Own vessel (RRT*)", "Target")


def simulate():
    """Run the Imazu case and return the track of each vessel.

    Returns:
        list: One (xs, ys) pair per vessel, own vessel first
    """
    runner = SimulationRunner()
    runner.imazu = True
    runner.imazu_number = IMAZU_CASE
    runner.os_algo = PLANNER
    runner.num_steps = NUM_STEPS
    runner.s = MAP_HALF_EXTENT_M
    runner.visu_figure = False
    runner.record_data = False
    vessels = runner.run()
    return [(list(v.trajectory[0]), list(v.trajectory[1])) for v in vessels]


def new_axes(tracks):
    """Create square axes that fit every track.

    Args:
        tracks (list): One (xs, ys) pair per vessel

    Returns:
        tuple: Figure and axes
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_IN)
    xs = [x for t in tracks for x in t[0]]
    ys = [y for t in tracks for y in t[1]]
    margin = 0.05 * max(max(xs) - min(xs), max(ys) - min(ys))
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Imazu case {IMAZU_CASE}, {PLANNER}")
    ax.grid(alpha=0.3)
    return fig, ax


def label_of(i):
    """Name a vessel for the legend.

    Args:
        i (int): Vessel index, own vessel is 0

    Returns:
        str: Legend label
    """
    if i < len(TRACK_LABELS):
        return TRACK_LABELS[i]
    return f"Target {i}"


def save_png(tracks, path):
    """Draw every full track with start and end markers.

    Args:
        tracks (list): One (xs, ys) pair per vessel
        path (str): Output PNG path
    """
    fig, ax = new_axes(tracks)
    for i, (xs, ys) in enumerate(tracks):
        line, = ax.plot(xs, ys, label=label_of(i))
        ax.plot(xs[0], ys[0], "o", color=line.get_color())
        ax.plot(xs[-1], ys[-1], ">", color=line.get_color())
    ax.legend(loc="best")
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_gif(tracks, path):
    """Animate the tracks step by step.

    Args:
        tracks (list): One (xs, ys) pair per vessel
        path (str): Output GIF path
    """
    fig, ax = new_axes(tracks)
    lines = [ax.plot([], [], label=label_of(i))[0] for i in range(len(tracks))]
    heads = [ax.plot([], [], "o", color=l.get_color())[0] for l in lines]
    ax.legend(loc="best")
    n_frames = min(len(t[0]) for t in tracks)

    # Each frame shows the tracks up to step k and a dot at the current position.
    def draw(k):
        """Draw frame k.

        Args:
            k (int): Step index

        Returns:
            list: Updated artists
        """
        for (xs, ys), line, head in zip(tracks, lines, heads):
            line.set_data(xs[:k + 1], ys[:k + 1])
            head.set_data([xs[k]], [ys[k]])
        return lines + heads

    anim = FuncAnimation(fig, draw, frames=n_frames, blit=True)
    anim.save(path, writer=PillowWriter(fps=GIF_FPS))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    tracks = simulate()
    save_png(tracks, os.path.join(OUT_DIR, "imazu2.png"))
    save_gif(tracks, os.path.join(OUT_DIR, "imazu2.gif"))
    print(f"wrote {OUT_DIR}/imazu2.png and {OUT_DIR}/imazu2.gif")
