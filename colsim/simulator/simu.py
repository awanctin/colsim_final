from .display.calcul_tools import *
from .display.draw import *
from .vessels.boat import Boat
from .vessels.whale import Whale
from .vessels.fisherman import Fisherman
from .vessels.island import Island
from .vessels.ship import Ship
from .vessels.agent import Agent
from ..planners.reactive.potential_fields import *
import csv
import re
import ast
from datetime import datetime
import os
from matplotlib.animation import FuncAnimation, PillowWriter
from .display.representation import Representation
import pytz

_SEA_OBJECT_CLASSES = {
    c.__name__: c for c in (Boat, Whale, Fisherman, Island, Ship, Agent)
}


class Simulation:
    def __init__(self, sea_objects, dt, k, directory=None):
        """Store the sea objects and the run settings.

        Args:
            sea_objects (list): All sea objects
            dt (float): Time step, s
            k (int): Start step index
            directory (str): Output folder, or None for saves/data
        """
        self.sea_objects = sea_objects
        self.dt = dt
        self.k = k
        timezone = pytz.timezone("Australia/Adelaide")
        self.save = datetime.now(timezone).strftime("%Y%m%d%H%M%S%f")
        self.directory = "saves/data" if directory is None else directory
        self.render_3D = True
        self.size_map = 0
        self.save_all = False
        self.observer = None
        self.verbose = False

    def run(self, record_data, num_steps, mmsi_list, rules, table, ax, ax_leg, Ɛ, s):
        """Move and draw every object for a number of steps.

        Args:
            record_data (bool): Whether to store the trajectory
            num_steps (int): Number of steps
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            ax_leg (matplotlib.axes.Axes): Legend axes, or None
            Ɛ (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
        """
        self.size_map = s
        for _ in range(num_steps):

            if ax is not None:
                clear(ax)

            for sea_objects in self.sea_objects:
                sea_objects.move(
                    record_data,
                    self.sea_objects,
                    mmsi_list,
                    rules,
                    table,
                    ax,
                    Ɛ,
                    s,
                    self.k,
                    self.dt,
                )

                if ax is not None:
                    sea_objects.draw(ax, Ɛ)

            if ax is not None:
                ax.set_xlim(-s, s)
                ax.set_ylim(-s, s)

            if ax_leg is not None:
                ax_leg.set_xlim(-s, s)
                ax_leg.set_ylim(-s, s)

    def run_with_data(self, record_data, num_steps, mmsi_list, rules, table, ax, Ɛ, s):
        """Run the simulation and write every state to a CSV file.

        Args:
            record_data (bool): Whether to store the trajectory
            num_steps (int): Number of steps
            mmsi_list (list): Identifiers of all sea objects
            rules (list): Active rules table
            table (matplotlib.table.Table): Rules table display, or None
            ax (matplotlib.axes.Axes): Axes to draw on, or None
            Ɛ (float): Safety margin added to the radius, m
            s (float): Half size of the map, m
        """
        directory = self.directory + "/data"
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(f"{directory}/{self.save}.csv", "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)

            for _step in range(num_steps):
                for sea_objects in self.sea_objects:
                    log_data = sea_objects.move(
                        record_data,
                        self.sea_objects,
                        mmsi_list,
                        rules,
                        table,
                        ax,
                        Ɛ,
                        s,
                        self.k,
                        self.dt,
                        self.verbose,
                    )

                    csv_writer.writerow(log_data)
                if self.observer is not None:
                    self.observer(_step, self.sea_objects)
        collided = next((o for o in self.sea_objects if o.collision_coordinates), None)
        if collided is not None or self.save_all:
            coords = collided.collision_coordinates if collided is not None else []
            if coords:
                print("Collision coordinates:", coords)
            col_x1, col_y1 = (coords[0][0], coords[0][1]) if coords else (0, 0)
            col_x2, col_y2 = (
                (coords[1][0], coords[1][1]) if len(coords) > 1 else (col_x1, col_y1)
            )
            sea_obj = self.process_data(f"{directory}/{self.save}.csv")
            with open(f"{directory}/sea_obj_{self.save}.txt", "w") as file:
                file.write(str(sea_obj))
            self.visualize_data(
                sea_obj,
                directory_attr=self.directory + "/plots",
                col_coordinates=(
                    [(col_x1, col_y1), (col_x2, col_y2)] if coords else None
                ),
            )
            if self.render_3D:
                rep = Representation()
                rep.create_3D_rep(sea_obj)
                if not os.path.exists("saves/3D"):
                    os.makedirs("saves/3D", exist_ok=True)
                rep.save(f"{self.save}.html", "saves/3D")
        else:
            os.remove(f"{directory}/{self.save}.csv")

    def process_data(self, log_data):
        """Read a saved CSV file into states per vessel.

        Args:
            log_data (str): Path to the CSV file

        Returns:
            dict: Recorded states per vessel
        """
        log_data = os.path.abspath(log_data)

        if not os.path.exists(log_data):
            raise FileNotFoundError(f"File {log_data} does not exist")

        if not os.access(log_data, os.R_OK):
            raise PermissionError(f"File {log_data} is not readable")

        with open(log_data, "r") as csvfile:
            csv_reader = csv.reader(csvfile)
            sea_objects = {}
            rows = list(csv_reader)
            for row in rows:
                mmsi = row[0] + "_" + row[1]
                if mmsi not in sea_objects:
                    sea_objects[mmsi] = []
                cleaned_string_self = re.sub(r"[\n\s]+", ",", row[2].strip())
                cleaned_string_self = cleaned_string_self.replace("[,", "[")
                if row[1] != "Island":
                    cleaned_string_history = re.sub(r"[\n\s]+", ",", row[4].strip())
                    cleaned_string_history = cleaned_string_history.replace("[,", "[")
                    cleaned_string_history = cleaned_string_history.replace(",]", "]")
                    cleaned_string_history = cleaned_string_history.replace(",,", ",")
                    cleaned_string_history = cleaned_string_history.replace("array", "")
                    cleaned_string_history = cleaned_string_history.replace(":,", ":")
                else:
                    cleaned_string_history = "None"
                sea_objects[mmsi].append(
                    [
                        row[1],
                        ast.literal_eval(cleaned_string_self),
                        row[3],
                        ast.literal_eval(cleaned_string_history),
                    ]
                )
            return sea_objects

    def visualize_data(
        self,
        all_sea_objects,
        directory_attr="saves/plots",
        other_alpha=1,
        add_title="",
        col_coordinates=None,
    ):
        """Plot every trajectory and save the figure.

        Args:
            all_sea_objects (dict): Recorded states per vessel
            directory_attr (str): Output folder
            other_alpha (float): Transparency of the other vessels
            add_title (str): Text added to the title
            col_coordinates (list): Collision points to mark, or None
        """
        plt.ioff()
        fig, ax = plt.subplots(figsize=(10, 8))
        list_color = []
        radius = {"Agent": 1, "Ship": 4, "Boat": 1, "Whale": 2, "Island": 4}.get(
            list(all_sea_objects.keys())[0].split("_")[1], 1
        )
        for key, value in all_sea_objects.items():
            x = [i[1][0] for i in value]
            y = [i[1][1] for i in value]
            a = ax.plot(
                x,
                y,
                "-",
                label=str(key),
                zorder=1,
                alpha=(1 if key.split("_")[1] == "Agent" else other_alpha),
            )
            list_color.append(a[0].get_color())

        i = 0
        for key in all_sea_objects.keys():
            processed_string = key.split("_")
            mmsi = processed_string[0]
            x_final = float(np.round(all_sea_objects[key][-1][1][0], 1))
            y_final = float(np.round(all_sea_objects[key][-1][1][1], 1))
            theta_final = float(all_sea_objects[key][-1][1][3])
            g_x = float(np.round(all_sea_objects[key][-1][1][4], 1))
            g_y = float(np.round(all_sea_objects[key][-1][1][5], 1))
            x_0 = float(np.round(all_sea_objects[key][0][1][0], 1))
            y_0 = float(np.round(all_sea_objects[key][0][1][1], 1))
            object = _SEA_OBJECT_CLASSES[processed_string[1]](
                int(processed_string[0]),
                x_final,
                y_final,
                0,
                theta_final,
                "APF",
                destination_distance=0,
                goal=(g_x, g_y),
            )
            color_goal = list_color[i]
            object.draw(
                ax, 0, alpha=(other_alpha if key.split("_")[1] != "Agent" else 1)
            )
            ax.annotate(
                f"{mmsi}",
                ((x_final, y_final) if key.split("_")[1] == "Island" else (g_x, g_y)),
                textcoords="offset points",
                xytext=(0, 0),
                ha="center",
            )
            i += 1

        directory = directory_attr
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        if col_coordinates:
            col_x1, col_y1 = col_coordinates[0]
            col_x2, col_y2 = col_coordinates[1]
            ax.plot(
                [col_x1, col_x2],
                [col_y1, col_y2],
                color="red",
                linestyle="--",
                linewidth=2,
                label="Collision Line",
            )
            ax.scatter(
                [col_x1, col_x2],
                [col_y1, col_y2],
                color="red",
                s=100,
                label="Collision Points",
            )
        max_x = max(
            [max([i[1][0] for i in value]) for value in all_sea_objects.values()]
        )
        max_x = max([float(max_x[0] + 1), self.size_map])

        min_x = min(
            [min([i[1][0] for i in value]) for value in all_sea_objects.values()]
        )
        min_x = min([float(min_x[0] - 1), -self.size_map])
        max_y = max(
            [max([i[1][1] for i in value]) for value in all_sea_objects.values()]
        )
        max_y = max([float(max_y[0] + 1), self.size_map])
        min_y = min(
            [min([i[1][1] for i in value]) for value in all_sea_objects.values()]
        )
        min_y = min([float(min_y[0] - 1), -self.size_map])
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Simulation Results \n" + "Reason for stopping: " + add_title)

        handles, labels = ax.get_legend_handles_labels()

        def get_numeric_prefix(label):
            """Read the identifier at the start of a label.

            Args:
                label (str): Label like "12_Ship"

            Returns:
                int: Identifier, or inf if missing
            """
            try:
                return int(label.split("_")[0])
            except (ValueError, IndexError):
                return float("inf")

        sorted_pairs = sorted(
            zip(handles, labels), key=lambda pair: get_numeric_prefix(pair[1])
        )

        sorted_handles, sorted_labels = zip(*sorted_pairs)
        ax.legend(
            sorted_handles,
            sorted_labels,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )

        plt.tight_layout()
        ax.set_aspect("equal", adjustable="box")
        plt.grid()
        plt.savefig(
            f"{directory}/plot_{self.save}_{add_title}.png", bbox_inches="tight"
        )
        plt.close(fig)

    def visualize_data_2(
        self,
        all_sea_objects,
        directory_attr="saves/plots",
        other_alpha=1,
        add_title="",
        algo="Default",
    ):
        """Plot every trajectory with a cleaner legend and save the figure.

        Args:
            all_sea_objects (dict): Recorded states per vessel
            directory_attr (str): Output folder
            other_alpha (float): Transparency of the other vessels
            add_title (str): Text added to the title
            algo (str): Algorithm name shown in the title
        """
        import matplotlib.pyplot as plt
        import os

        plt.ioff()
        fig, ax = plt.subplots(figsize=(10, 8))

        def get_radius(obj_type):
            """Give the drawing radius of an object type.

            Args:
                obj_type (str): Object type name

            Returns:
                int: Radius
            """
            return {"Agent": 1, "Ship": 4, "Boat": 1, "Whale": 2, "Island": 4}.get(
                obj_type, 1
            )

        handles = []
        labels = []

        for key, value in all_sea_objects.items():
            obj_type = key.split("_")[1]
            x = [i[1][0] for i in value]
            y = [i[1][1] for i in value]

            label = f"{key} ({algo})"
            (line,) = ax.plot(
                x,
                y,
                "-",
                label=label,
                zorder=1,
                alpha=(1 if obj_type == "Agent" else other_alpha),
            )
            handles.append(line)
            labels.append(label)

            x_final, y_final = x[-1], y[-1]
            r = get_radius(obj_type)
            goal = plt.Circle(
                (x_final, y_final), r, color=line.get_color(), fill=False, alpha=0.5
            )
            ax.add_artist(goal)
            ax.annotate(
                key.split("_")[0],
                (x_final, y_final),
                textcoords="offset points",
                xytext=(0, 0),
                ha="center",
            )

        ax.set_xlim(-self.size_map, self.size_map)
        ax.set_ylim(-self.size_map, self.size_map)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Simulation Results \n" + "Reason for stopping: " + add_title)

        sorted_pairs = sorted(
            zip(handles, labels), key=lambda pair: int(pair[1].split("_")[0])
        )
        sorted_handles, sorted_labels = zip(*sorted_pairs)
        ax.legend(
            sorted_handles,
            sorted_labels,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
        plt.tight_layout()
        plt.grid()

        directory = directory_attr
        os.makedirs(directory, exist_ok=True)
        plt.savefig(
            f"{directory}/plot_{self.save}_{add_title}.png", bbox_inches="tight"
        )
        plt.close(fig)

    def use_history(self, sea_object, all_sea_objects):
        """Collect the encounters of one vessel during collisions.

        Args:
            sea_object (SeaObject): Vessel to look at
            all_sea_objects (dict): Recorded states per vessel

        Returns:
            dict: Encounter points per vessel
        """
        mmsi = sea_object.mmsi
        name = sea_object.name
        key = f"{mmsi}_{name}"
        col = {}
        if key in all_sea_objects:
            for i in range(len(all_sea_objects[key])):
                current_status = all_sea_objects[key][i][1]
                collision = all_sea_objects[key][i][2]
                history = all_sea_objects[key][i][3]
                if history is not None:
                    history_key = history.keys()
                    if float(collision):
                        for h_key in history_key:
                            if key not in col:
                                col[key] = {}
                            if h_key not in col[key]:
                                col[key][h_key] = {
                                    "current status": [current_status],
                                    "id": [h_key],
                                    "x": [],
                                    "y": [],
                                    "v": [],
                                    "theta": [],
                                }
                            for k in range(len(history[h_key])):
                                for l in range(len(history[h_key][k])):
                                    col[key][h_key]["x"].append(
                                        history[h_key][k][l][0][0]
                                    )
                                    col[key][h_key]["y"].append(
                                        history[h_key][k][l][0][1]
                                    )
                                    col[key][h_key]["v"].append(
                                        history[h_key][k][l][0][2]
                                    )
                                    col[key][h_key]["theta"].append(
                                        history[h_key][k][l][0][3]
                                    )
        self.col = col
        return col

    def show_history(self, col):
        """Plot and save the encounter points of each vessel.

        Args:
            col (dict): Encounter points per vessel
        """
        fig, ax = plt.subplots()
        directory = "saves/test_plots"
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        for key in col:
            for k in col[key]:
                encounter = col[key][k]["id"]
                ax.scatter(
                    col[key][k]["x"], col[key][k]["y"], label=encounter, marker="x"
                )
            ax.set_title(f"Encounters of {key}")
            ax.legend()
            plt.savefig(f"{directory}/history_{self.save}_{key}_{encounter}.png")
        plt.close(fig)

    def show_animated_vectors(self):
        """Animate the speed vectors of the stored encounters."""
        if not hasattr(self, "col") or not self.col:
            print("No collision data to display.")
            return

        fig, ax = plt.subplots()
        xs = [
            x for key in self.col for k in self.col[key] for x in self.col[key][k]["x"]
        ]
        ys = [
            y for key in self.col for k in self.col[key] for y in self.col[key][k]["y"]
        ]
        if xs:
            ax.set_xlim(min(xs), max(xs))
            ax.set_ylim(min(ys), max(ys))
        directory = "saves/test_plots_vectors"
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        m_len = 0
        for key in self.col:
            for k in self.col[key]:
                m_len = max(m_len, len(self.col[key][k]["x"]))

        if m_len == 0:
            print("No data to animate.")
            return

        print("max_len =", m_len)

        ani = FuncAnimation(
            fig,
            self.update_rules,
            frames=np.arange(0, m_len),
            interval=100,
            fargs=(ax,),
            init_func=self.init_plot,
        )
        writer = PillowWriter(fps=20)
        ani.save(f"{directory}/history_{self.save}.gif", writer=writer)
        plt.close(fig)

    def init_plot(self):
        """Create an empty figure for the animation.

        Returns:
            matplotlib.axes.Axes: New axes
        """
        fig, ax = plt.subplots()
        ax.set_title("Initial Plot")
        return ax

    def update_rules(self, frame, ax):
        """Draw the speed vectors of one animation frame.

        Args:
            frame (int): Frame index
            ax (matplotlib.axes.Axes): Axes to draw on

        Returns:
            matplotlib.axes.Axes: Updated axes
        """
        ax.clear()
        for key in self.col:
            for k in self.col[key]:
                if frame < len(self.col[key][k]["x"]):
                    encounter = self.col[key][k]["id"]
                    v = self.col[key][k]["v"][frame]
                    theta = self.col[key][k]["theta"][frame]
                    ax.quiver(
                        self.col[key][k]["x"][frame],
                        self.col[key][k]["y"][frame],
                        v * np.cos(theta),
                        v * np.sin(theta),
                    )
            ax.set_title(f"Encounters of {key}")
        return ax
