from .display.calcul_tools import *
from .display.draw import *
from .simu import Simulation
from .vessels.boat import Boat
from .vessels.whale import Whale
from .vessels.fisherman import Fisherman
from .vessels.island import Island
from .vessels.ship import Ship
from .vessels.agent import Agent
from .spatial_grid import SpatialGrid
import random

OBJ_TYPES = ["Whale", "Boat", "Ship", "Island"]


def dist_squared(x1, y1, x2, y2):
    """Compute the squared distance between two points.

    Args:
        x1 (float): First x
        y1 (float): First y
        x2 (float): Second x
        y2 (float): Second y

    Returns:
        float: Squared distance
    """
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


class SimulationRunner:
    def __init__(self):
        """Set the default map, timing, rules table and display options."""
        self.s = 15000
        self.dt = 10
        self.k = 0.5
        self.Ɛ = 20
        self.environment = 2
        self.num_steps = 100
        self.record_data = False
        self.rules = [["Rules :"], ["Finish OT"], ["Left OT"], ["Right OT"], ["R to R"]]
        self.legend = [
            ["Finish overtaking the obstacle", "Finish OT"],
            ["Overtaking the obstacle on the left side", "Left OT"],
            ["Overtaking the obstacle on the right side", "Right OT"],
            ["Red to red rule to avoid the collision", "R to R"],
        ]
        self.list_sea_objects = []
        self.visu_figure = True
        self.agent = False
        self.imazu = False
        self.imazu_number = 0
        self.min_goal_distance = 500.0
        self.random_v_min = 5.0
        self.random_v_max = 15.0
        self.min_object_distance = 200.0
        self.goal_clearance = 100.0
        self.generate_from_file = False
        self.directory = "data"
        self.save_all_runner = False
        self.verbose = False
        self.seed = None
        self.observer = None
        self.render_3D = False

        self.os_algo = "RRT*"
        self.hybrid_certificate = False
        self.hybrid_policy = None
        self.hybrid_gate = 500.0
        self.hybrid_horizon = None
        self.hybrid_bind_yaw = True
        self.hybrid_trigger_k = None
        self.hybrid_release_lookahead = None
        self.hybrid_veto_margin = None
        self.hybrid_action_shield = None
        self.hybrid_shield_margin = None
        self.hybrid_shield_pick = None
        self.hybrid_margin_steps = None
        self.hybrid_n_track = None
        self.hybrid_speed_action = None
        self.hybrid_alpha_col_release = None
        self.hybrid_rule17b_relax = None
        self.hybrid_p_switch = None
        self.hybrid_max_emergency_steps = None
        self.hybrid_bin_fields = None
        self.enable_speed_control: bool = False
        self.enable_reverse: bool = False

    def initialize_sea_objects(self):
        """Build one of the fixed test environments.

        Returns:
            list: Sea objects of the chosen environment
        """
        sea_objects = []

        if self.environment == 0:
            sea_objects.append(Whale(111, -0.5, -3.5, 1.5, 1.5, "APF"))
            sea_objects.append(Boat(222, -3, 5, 1.5, 0.25, "APF"))
            sea_objects.append(Island(333, 0, 2, 0, 1))
            sea_objects.append(Ship(444, 10, -4, 1.5, 3, "APF"))

        elif self.environment == 1:
            """Only one ship, has to go straight."""
            sea_objects.append(Ship(444, 10, -4, 1.5, pi * 7 / 8, "PSO"))

        elif self.environment == 2:
            """One island to slightly contourn."""
            sea_objects.append(Island(333, 0, 2, 0, 1))
            sea_objects.append(Ship(444, 10, -4, 1.5, pi * 7 / 8, "PSO"))

        elif self.environment == 3:
            """U shape with a potential minimum."""
            sea_objects.append(Island(333, 0, 2, 0, 1))
            sea_objects.append(Island(332, 0, -6, 0, 1))
            sea_objects.append(Ship(444, 10, -4, 1.5, 3, "APF"))

        elif self.environment == 4:
            """Slalom with 2 islands and 1 ship."""
            sea_objects.append(Ship(444, 10, -10, 1.5, 2.4, "PSO", 30))
            sea_objects.append(Island(333, -2, 7, 0, 1))
            sea_objects.append(Island(332, 2, -10, 0, 1))

        elif self.environment == 5:
            """Two boats facing each other."""
            sea_objects.append(Ship(444, 12, 0, 1.5, pi, "PSO", 30))
            sea_objects.append(Ship(555, -12, 0, 1.5, 0, "PSO", 30))

        elif self.environment == 6:
            """A ship overtaking a boat."""
            real = True
            if real == False:
                sea_objects.append(Ship(444, -18, 0, 3, 0, "PSO", 36))
                sea_objects.append(Ship(555, -7, 0, 1, 0, "PSO", 15))
            else:
                sea_objects.append(Ship(444, -15, -2, 3, 0.4, "ACO", 32))
                sea_objects.append(Ship(555, -5, 0, 1, 0, "ACO", 15))

        else:
            """Not a fixed env, to be changed."""
            sea_objects.append(Ship(444, 12, 0, 1.5, 3.14, "ACO", 30))
            sea_objects.append(Island(333, 5, 2, 0, 1))
        return sea_objects

    def place_random_objects(self, object_type, spatial_grid, max_attempts=100):
        """Try to place one object and its goal at a free random spot.

        Args:
            object_type (str): Object type name
            spatial_grid (SpatialGrid): Grid of objects already placed
            max_attempts (int): Number of tries

        Returns:
            SeaObject: Placed object, or None if no free spot was found
        """
        self.max_attempts = max_attempts
        for _ in range(max_attempts):
            x = random.uniform(-self.s + self.Ɛ, self.s - self.Ɛ)
            y = random.uniform(-self.s + self.Ɛ, self.s - self.Ɛ)
            mmsi = random.randint(100, 999)
            valid_goal = False
            if object_type != "Island":
                for _ in range(max_attempts):
                    goal_x = random.uniform(-self.s + self.Ɛ, self.s - self.Ɛ)
                    goal_y = random.uniform(-self.s + self.Ɛ, self.s - self.Ɛ)
                    if (
                        np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)
                        >= self.min_goal_distance
                    ):
                        valid_goal = True
                        break
                if not valid_goal:
                    continue
            else:
                goal_x, goal_y = x, y
            v = random.uniform(self.random_v_min, self.random_v_max)
            theta = random.uniform(-np.pi, np.pi)
            temp_obj = self.create_temp_object(
                object_type,
                x,
                y,
                goal_x,
                goal_y,
                mmsi,
                self.os_algo,
                v,
                theta,
            )
            if self.check_position_and_goal(temp_obj, spatial_grid):
                spatial_grid.add_object(temp_obj)
                return temp_obj
        return None

    def initialize_sea_objects_random(
        self,
        list_sea_objects=None,
        previous_list_sea_objects=None,
    ):
        """Place random objects, or rebuild a previous list.

        Args:
            list_sea_objects (list): Object type names to place
            previous_list_sea_objects (list): Objects to copy, or None

        Returns:
            list: Placed sea objects
        """
        spatial_grid = SpatialGrid(cell_size=2 * self.Ɛ, map_size=self.s)
        sea_objects = []
        if previous_list_sea_objects:
            for i in range(len(previous_list_sea_objects)):

                obj = self.create_temp_object(
                    previous_list_sea_objects[i].name,
                    previous_list_sea_objects[i].x,
                    previous_list_sea_objects[i].y,
                    previous_list_sea_objects[i].goal_x,
                    previous_list_sea_objects[i].goal_y,
                    previous_list_sea_objects[i].mmsi,
                    previous_list_sea_objects[i].algo,
                    previous_list_sea_objects[i].v,
                    previous_list_sea_objects[i].theta,
                    previous_list_sea_objects[i].record_path,
                )
                if self.check_position_and_goal(obj, spatial_grid) or True:
                    spatial_grid.add_object(obj)
                    sea_objects.append(obj)
                else:
                    print(
                        f"Failed to place previous object {i.mmsi} after {self.max_attempts} attempts"
                    )
        if list_sea_objects:
            for _ in range(len(list_sea_objects)):
                obj = self.place_random_objects(list_sea_objects[_], spatial_grid)
                if obj:
                    sea_objects.append(obj)
                else:
                    print(f"Failed to place object after {self.max_attempts} attempts")
        return sea_objects

    def _check_position(self, x, y, radius, spatial_grid):
        """Check that a point is far enough from every nearby object and goal.

        Args:
            x (float): Point x, m
            y (float): Point y, m
            radius (float): Radius of the new object, m
            spatial_grid (SpatialGrid): Grid of objects already placed

        Returns:
            bool: True if the spot is free
        """
        nearby_objects = []
        cells = spatial_grid.get_relevant_cells(x, y, radius)
        for cell in cells:
            nearby_objects.extend(spatial_grid.grid.get(cell, []))

        return not any(
            (dist_squared(x, y, o.x, o.y) <= (radius + o.r + self.Ɛ) ** 2)
            or (dist_squared(x, y, o.goal_x, o.goal_y) <= (radius + o.r + self.Ɛ) ** 2)
            for o in nearby_objects
        )

    def check_position_and_goal(self, obj, spatial_grid):
        """Check that an object and its goal are both on free spots.

        Args:
            obj (SeaObject): Object to check
            spatial_grid (SpatialGrid): Grid of objects already placed

        Returns:
            bool: True if both are free
        """
        pos_valid = self._check_position(obj.x, obj.y, obj.r, spatial_grid)

        goal_valid = self._check_position(obj.goal_x, obj.goal_y, obj.r, spatial_grid)

        return pos_valid and goal_valid

    def create_temp_object(
        self,
        object_type,
        x,
        y,
        goal_x=None,
        goal_y=None,
        mmsi=None,
        algo=None,
        v=None,
        theta=None,
        record_path="plots",
    ):
        """Create a sea object of the given type.

        Args:
            object_type (str): Object type name
            x (float): Start x, m
            y (float): Start y, m
            goal_x (float): Goal x, m
            goal_y (float): Goal y, m
            mmsi (int): Vessel identifier
            algo (str): Name of the path planning algorithm
            v (float): Start speed, m/s
            theta (float): Start heading, rad
            record_path (str): Folder for saved plots

        Returns:
            SeaObject: New object
        """
        if object_type == "Whale":
            return Whale(mmsi, x, y, v, theta, algo=algo, goal=[goal_x, goal_y])
        elif object_type == "Boat":
            return Boat(
                mmsi,
                x,
                y,
                v,
                theta,
                algo=algo,
                goal=[goal_x, goal_y],
                record_path=record_path,
            )
        elif object_type == "Ship":
            return Ship(mmsi, x, y, v, theta, algo=algo, goal=[goal_x, goal_y])
        elif object_type == "Island":
            return Island(mmsi, x, y, 0, 0, goal=[x, y])
        elif object_type == "Agent":
            return Agent(
                mmsi, x, y, v, theta, algo=algo or "APF", goal=[goal_x, goal_y]
            )
        else:
            raise ValueError("Unknown object type: ", object_type)

    def random_object_list(self, nb):
        """Draw a random list of object types.

        Args:
            nb (int): Number of objects to draw

        Returns:
            list: Object type names, with the agent first if enabled
        """
        object_list = []
        if self.agent:
            object_list.append("Agent")
        for i in range(nb):
            object_list.append(random.choice(["Whale", "Boat", "Ship", "Island"]))
        return object_list

    def initialize_data(self, sea_object, rules):
        """Add one column per object to the rules table.

        Args:
            sea_object (list): All sea objects
            rules (list): Rules table

        Returns:
            tuple: Updated rules table and list of identifiers
        """
        mmsi_list = []
        for row in range(1, len(rules)):
            for col in range(len(sea_object)):
                rules[row].append(" ")
        for col in range(len(sea_object)):
            rules[0].append(sea_object[col].mmsi)
            mmsi_list.append(sea_object[col].mmsi)
        return rules, mmsi_list

    def imazu_situation(self, number):
        """Build one of the 22 Imazu test encounters.

        Args:
            number (int): Imazu case number

        Returns:
            list: Own vessel first, then the targets
        """
        KTS = 10 / 1.94384
        KM = 1000

        sea_objects = []
        sea_objects.append(
            Boat(000, 0, 0, 2.5 * KTS, 0, self.os_algo, goal=[20 * KM, 0])
        )

        if number == 1:
            sea_objects.append(
                Boat(
                    101,
                    10 * KM,
                    0,
                    1 * KTS,
                    180 * np.pi / 180,
                    "RRT*",
                    goal=[-2 * KM, 0],
                )
            )

        elif number == 2:
            sea_objects.append(
                Boat(
                    102,
                    5 * KM,
                    -2 * KM,
                    1 * KTS,
                    90 * np.pi / 180,
                    "RRT*",
                    goal=[5 * KM, 10 * KM],
                )
            )

        elif number == 3:
            sea_objects.append(
                Boat(103, 3 * KM, 0, 1 * KTS, 0, "RRT*", goal=[20 * KM, 0])
            )

        elif number == 4:
            sea_objects.append(
                Boat(
                    104,
                    3.44 * KM,
                    1.5 * KM,
                    1 * KTS,
                    295 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3.44 * KM + np.cos(295 * np.pi / 180) * 20 * KM,
                        1.5 * KM + np.sin(295 * np.pi / 180) * 20 * KM,
                    ],
                )
            )

        elif number == 5:
            sea_objects.append(
                Boat(
                    105,
                    5 * KM,
                    -2.01 * KM,
                    1 * KTS,
                    90 * np.pi / 180,
                    "RRT*",
                    goal=[5 * KM, 10 * KM],
                )
            )
            sea_objects.append(
                Boat(
                    106,
                    6.95 * KM,
                    0,
                    1 * KTS,
                    180 * np.pi / 180,
                    "RRT*",
                    goal=[-2 * KM, 0],
                )
            )

        elif number == 6:
            sea_objects.append(
                Boat(
                    107,
                    3.4 * KM,
                    -1.35 * KM,
                    1 * KTS,
                    45 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3.4 * KM + np.cos(45 * np.pi / 180) * 20 * KM,
                        -1.35 * KM + np.sin(45 * np.pi / 180) * 20 * KM,
                    ],
                )
            )
            sea_objects.append(
                Boat(
                    108,
                    3 * KM,
                    -0.35 * KM,
                    1 * KTS,
                    10 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3 * KM + np.cos(10 * np.pi / 180) * 20 * KM,
                        -0.35 * KM + np.sin(10 * np.pi / 180) * 20 * KM,
                    ],
                )
            )

        elif number == 7:
            sea_objects.append(
                Boat(109, 3 * KM, 0, 1 * KTS, 0, "RRT*", goal=[20 * KM, 0])
            )
            sea_objects.append(
                Boat(
                    110,
                    3.4 * KM,
                    -1.35 * KM,
                    1 * KTS,
                    45 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3.4 * KM + np.cos(45 * np.pi / 180) * 20 * KM,
                        -1.35 * KM + np.sin(45 * np.pi / 180) * 20 * KM,
                    ],
                )
            )

        elif number == 8:
            sea_objects.append(
                Boat(
                    111,
                    5 * KM,
                    -2.01 * KM,
                    1 * KTS,
                    90 * np.pi / 180,
                    "RRT*",
                    goal=[5 * KM, 10 * KM],
                )
            )
            sea_objects.append(
                Boat(
                    112,
                    7 * KM,
                    0,
                    1 * KTS,
                    180 * np.pi / 180,
                    "RRT*",
                    goal=[-2 * KM, 0],
                )
            )

        elif number == 9:
            sea_objects.append(
                Boat(
                    113,
                    3.4 * KM,
                    -1.34 * KM,
                    1 * KTS,
                    45 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3.4 * KM + np.cos(45 * np.pi / 180) * 20 * KM,
                        -1.34 * KM + np.sin(45 * np.pi / 180) * 20 * KM,
                    ],
                )
            )
            sea_objects.append(
                Boat(
                    114,
                    5 * KM,
                    -2.0 * KM,
                    1 * KTS,
                    90 * np.pi / 180,
                    "RRT*",
                    goal=[5 * KM, 10 * KM],
                )
            )

        elif number == 10:
            sea_objects.append(
                Boat(
                    115,
                    3 * KM,
                    0.35 * KM,
                    1 * KTS,
                    350 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3 * KM + np.cos(350 * np.pi / 180) * 20 * KM,
                        0.35 * KM + np.sin(350 * np.pi / 180) * 20 * KM,
                    ],
                )
            )
            sea_objects.append(
                Boat(
                    116,
                    4.4 * KM,
                    -1.78 * KM,
                    1 * KTS,
                    90 * np.pi / 180,
                    "RRT*",
                    goal=[4.4 * KM, 10 * KM],
                )
            )

        elif number == 11:
            sea_objects.append(
                Boat(
                    117,
                    5 * KM,
                    2.0 * KM,
                    1 * KTS,
                    -90 * np.pi / 180,
                    "RRT*",
                    goal=[5 * KM, -10 * KM],
                )
            )
            sea_objects.append(
                Boat(
                    118,
                    3.4 * KM,
                    -1.35 * KM,
                    1 * KTS,
                    45 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3.4 * KM + np.cos(45 * np.pi / 180) * 20 * KM,
                        -1.35 * KM + np.sin(45 * np.pi / 180) * 20 * KM,
                    ],
                )
            )

        elif number == 12:
            sea_objects.append(
                Boat(
                    119,
                    7 * KM,
                    0,
                    1 * KTS,
                    180 * np.pi / 180,
                    "RRT*",
                    goal=[-2 * KM, 0],
                )
            )
            sea_objects.append(
                Boat(
                    120,
                    3 * KM,
                    0.35 * KM,
                    1 * KTS,
                    -10 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3 * KM + np.cos(-10 * np.pi / 180) * 20 * KM,
                        0.35 * KM + np.sin(-10 * np.pi / 180) * 20 * KM,
                    ],
                )
            )
            sea_objects.append(
                Boat(
                    121,
                    3.44 * KM,
                    -1.35 * KM,
                    1 * KTS,
                    45 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3.44 * KM + np.cos(45 * np.pi / 180) * 20 * KM,
                        -1.35 * KM + np.sin(45 * np.pi / 180) * 20 * KM,
                    ],
                )
            )

        elif number == 13:
            sea_objects.append(
                Boat(
                    122,
                    6 * KM,
                    0,
                    1 * KTS,
                    180 * np.pi / 180,
                    "RRT*",
                    goal=[-2 * KM, 0],
                )
            )
            sea_objects.append(
                Boat(
                    123,
                    3 * KM,
                    0.35 * KM,
                    1 * KTS,
                    350 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3 * KM + np.cos(350 * np.pi / 180) * 20 * KM,
                        0.35 * KM + np.sin(350 * np.pi / 180) * 20 * KM,
                    ],
                )
            )
            sea_objects.append(
                Boat(
                    124,
                    3.4 * KM,
                    1.47 * KM,
                    1 * KTS,
                    295 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3.4 * KM + np.cos(295 * np.pi / 180) * 20 * KM,
                        1.47 * KM + np.sin(295 * np.pi / 180) * 20 * KM,
                    ],
                )
            )

        elif number == 14:
            sea_objects.append(
                Boat(
                    125,
                    3.4 * KM,
                    -1.35 * KM,
                    1 * KTS,
                    45 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3.4 * KM + np.cos(45 * np.pi / 180) * 20 * KM,
                        -1.35 * KM + np.sin(45 * np.pi / 180) * 20 * KM,
                    ],
                )
            )
            sea_objects.append(
                Boat(
                    126,
                    3 * KM,
                    -0.35 * KM,
                    1 * KTS,
                    10 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3 * KM + np.cos(10 * np.pi / 180) * 20 * KM,
                        -0.35 * KM + np.sin(10 * np.pi / 180) * 20 * KM,
                    ],
                )
            )
            sea_objects.append(
                Boat(
                    127,
                    5 * KM,
                    -2.0 * KM,
                    1 * KTS,
                    90 * np.pi / 180,
                    "RRT*",
                    goal=[5 * KM, 10 * KM],
                )
            )

        elif number == 15:
            sea_objects.append(
                Boat(128, 3 * KM, 0, 1 * KTS, 0, "RRT*", goal=[20 * KM, 0])
            )
            sea_objects.append(
                Boat(
                    129,
                    3.4 * KM,
                    -1.35 * KM,
                    1 * KTS,
                    45 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3.4 * KM + np.cos(45 * np.pi / 180) * 20 * KM,
                        -1.35 * KM + np.sin(45 * np.pi / 180) * 20 * KM,
                    ],
                )
            )
            sea_objects.append(
                Boat(
                    130,
                    5 * KM,
                    -2.0 * KM,
                    1 * KTS,
                    90 * np.pi / 180,
                    "RRT*",
                    goal=[5 * KM, 10 * KM],
                )
            )

        elif number == 16:
            sea_objects.append(
                Boat(
                    131,
                    3.4 * KM,
                    1.35 * KM,
                    1 * KTS,
                    -45 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3.4 * KM + np.cos(-45 * np.pi / 180) * 20 * KM,
                        1.35 * KM + np.sin(-45 * np.pi / 180) * 20 * KM,
                    ],
                )
            )
            sea_objects.append(
                Boat(
                    132,
                    5 * KM,
                    2.0 * KM,
                    1 * KTS,
                    -90 * np.pi / 180,
                    "RRT*",
                    goal=[5 * KM, -10 * KM],
                )
            )
            sea_objects.append(
                Boat(
                    133,
                    5 * KM,
                    -2.0 * KM,
                    1 * KTS,
                    90 * np.pi / 180,
                    "RRT*",
                    goal=[5 * KM, 10 * KM],
                )
            )

        elif number == 17:
            sea_objects.append(
                Boat(134, 3 * KM, 0, 1 * KTS, 0, "RRT*", goal=[20 * KM, 0])
            )
            sea_objects.append(
                Boat(
                    135,
                    3 * KM,
                    0.35 * KM,
                    1 * KTS,
                    -10 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3 * KM + np.cos(-10 * np.pi / 180) * 20 * KM,
                        0.35 * KM + np.sin(-10 * np.pi / 180) * 20 * KM,
                    ],
                )
            )
            sea_objects.append(
                Boat(
                    136,
                    3.4 * KM,
                    -1.34 * KM,
                    1 * KTS,
                    45 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3.4 * KM + np.cos(45 * np.pi / 180) * 20 * KM,
                        -1.34 * KM + np.sin(45 * np.pi / 180) * 20 * KM,
                    ],
                )
            )

        elif number == 18:
            sea_objects.append(
                Boat(
                    137,
                    3.3 * KM,
                    -0.4 * KM,
                    1 * KTS,
                    10 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3.3 * KM + np.cos(10 * np.pi / 180) * 20 * KM,
                        -0.4 * KM + np.sin(10 * np.pi / 180) * 20 * KM,
                    ],
                )
            )
            sea_objects.append(
                Boat(
                    138,
                    3.4 * KM,
                    -1.35 * KM,
                    1 * KTS,
                    45 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3.4 * KM + np.cos(45 * np.pi / 180) * 20 * KM,
                        -1.35 * KM + np.sin(45 * np.pi / 180) * 20 * KM,
                    ],
                )
            )
            sea_objects.append(
                Boat(
                    139,
                    6.5 * KM,
                    -1.47 * KM,
                    1 * KTS,
                    135 * np.pi / 180,
                    "RRT*",
                    goal=[
                        6.5 * KM + np.cos(135 * np.pi / 180) * 20 * KM,
                        -1.47 * KM + np.sin(135 * np.pi / 180) * 20 * KM,
                    ],
                )
            )

        elif number == 19:
            sea_objects.append(
                Boat(
                    140,
                    3 * KM,
                    -0.35 * KM,
                    1 * KTS,
                    10 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3 * KM + np.cos(10 * np.pi / 180) * 20 * KM,
                        -0.35 * KM + np.sin(10 * np.pi / 180) * 20 * KM,
                    ],
                )
            )
            sea_objects.append(
                Boat(
                    141,
                    3 * KM,
                    0.35 * KM,
                    1 * KTS,
                    -10 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3 * KM + np.cos(-10 * np.pi / 180) * 20 * KM,
                        0.35 * KM + np.sin(-10 * np.pi / 180) * 20 * KM,
                    ],
                )
            )
            sea_objects.append(
                Boat(
                    142,
                    6.5 * KM,
                    -1.42 * KM,
                    1 * KTS,
                    135 * np.pi / 180,
                    "RRT*",
                    goal=[
                        6.5 * KM + np.cos(135 * np.pi / 180) * 20 * KM,
                        -1.42 * KM + np.sin(135 * np.pi / 180) * 20 * KM,
                    ],
                )
            )

        elif number == 20:
            sea_objects.append(
                Boat(143, 3 * KM, 0, 1 * KTS, 0, "RRT*", goal=[20 * KM, 0])
            )
            sea_objects.append(
                Boat(
                    144,
                    3 * KM,
                    -0.35 * KM,
                    1 * KTS,
                    10 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3 * KM + np.cos(10 * np.pi / 180) * 20 * KM,
                        -0.35 * KM + np.sin(10 * np.pi / 180) * 20 * KM,
                    ],
                )
            )
            sea_objects.append(
                Boat(
                    145,
                    4.4 * KM,
                    -1.78 * KM,
                    1 * KTS,
                    90 * np.pi / 180,
                    "RRT*",
                    goal=[4.4 * KM, 10 * KM],
                )
            )

        elif number == 21:
            sea_objects.append(
                Boat(
                    146,
                    2.7 * KM,
                    -0.35 * KM,
                    1 * KTS,
                    10 * np.pi / 180,
                    "RRT*",
                    goal=[
                        2.7 * KM + np.cos(10 * np.pi / 180) * 20 * KM,
                        -0.35 * KM + np.sin(10 * np.pi / 180) * 20 * KM,
                    ],
                )
            )
            sea_objects.append(
                Boat(
                    147,
                    2.7 * KM,
                    0.32 * KM,
                    1 * KTS,
                    -10 * np.pi / 180,
                    "RRT*",
                    goal=[
                        2.7 * KM + np.cos(-10 * np.pi / 180) * 20 * KM,
                        0.32 * KM + np.sin(-10 * np.pi / 180) * 20 * KM,
                    ],
                )
            )
            sea_objects.append(
                Boat(
                    148,
                    4.4 * KM,
                    -1.78 * KM,
                    1 * KTS,
                    90 * np.pi / 180,
                    "RRT*",
                    goal=[4.4 * KM, 10 * KM],
                )
            )

        elif number == 22:
            sea_objects.append(
                Boat(149, 3 * KM, 0, 1 * KTS, 0, "RRT*", goal=[20 * KM, 0])
            )
            sea_objects.append(
                Boat(
                    150,
                    3.94 * KM,
                    -1.73 * KM,
                    1 * KTS,
                    45 * np.pi / 180,
                    "RRT*",
                    goal=[
                        3.94 * KM + np.cos(45 * np.pi / 180) * 20 * KM,
                        -1.73 * KM + np.sin(45 * np.pi / 180) * 20 * KM,
                    ],
                )
            )
            sea_objects.append(
                Boat(
                    151,
                    5 * KM,
                    -2.0 * KM,
                    1 * KTS,
                    90 * np.pi / 180,
                    "RRT*",
                    goal=[5 * KM, 10 * KM],
                )
            )

        return sea_objects

    def run(self):
        """Build the scene, run the simulation and save the outputs.

        Returns:
            list: Sea objects at the end of the run
        """
        if self.seed is not None:
            import random
            import numpy as _np

            _s = int(self.seed) % (2**32)
            random.seed(_s)
            _np.random.seed(_s)
        if self.imazu:
            sea_objects = self.imazu_situation(self.imazu_number)
        elif self.generate_from_file:
            sea_objects = self.list_sea_objects
        else:
            sea_objects = self.initialize_sea_objects_random(self.list_sea_objects)

        if self.enable_speed_control and sea_objects:
            from ..control.speed_governor import SpeedGovernor

            own_ship = sea_objects[0]
            own_ship._speed_gov = SpeedGovernor(
                v_nominal=own_ship.v,
                enable_reverse=self.enable_reverse,
            )
        if self.hybrid_certificate and sea_objects:
            own_ship = sea_objects[0]
            own_ship.hybrid_certificate = True
            own_ship.hybrid_policy = self.hybrid_policy
            own_ship.hybrid_gate = self.hybrid_gate
            own_ship.hybrid_horizon = self.hybrid_horizon
            for knob in (
                "trigger_k",
                "release_lookahead",
                "veto_margin",
                "action_shield",
                "shield_margin",
                "shield_pick",
                "margin_steps",
                "n_track",
                "speed_action",
                "alpha_col_release",
                "rule17b_relax",
                "p_switch",
                "max_emergency_steps",
                "bin_fields",
            ):
                value = getattr(self, f"hybrid_{knob}")
                if value is not None:
                    setattr(own_ship, f"hybrid_{knob}", value)
            if own_ship.algo != "hybrid":
                raise ValueError(
                    f"hybrid_certificate=True needs the own-ship on algo='hybrid', "
                    f"got {own_ship.algo!r} — set os_algo='hybrid' (Imazu) or build the "
                    f"own-ship with algo='hybrid'. Silently running the compliant planner "
                    f"while reporting a hybrid run is the failure this check exists to stop."
                )
            if self.hybrid_bind_yaw and own_ship.v > 0:
                from ..certificate.cert_switch import R_MIN_M

                own_ship.max_rot_rad_s = own_ship.v / R_MIN_M

        for obj in sea_objects:
            obj.sea_objects = sea_objects
            for other in obj.sea_objects:
                if obj != other:
                    situation = obj.COLREGs_situation(other)
                    obj.list_colregs.append([obj.mmsi, other.mmsi, situation])
        rules, mmsi_list = self.initialize_data(sea_objects, self.rules)
        simulation = Simulation(sea_objects, self.dt, self.k, self.directory)
        simulation.render_3D = self.render_3D
        simulation.agent = self.agent
        simulation.size_map = self.s
        simulation.save_all = self.save_all_runner
        simulation.verbose = self.verbose
        simulation.observer = self.observer

        if self.visu_figure:
            fig, ax = init_figure(-self.s, self.s, -self.s, self.s)
            fig_leg, ax_leg = init_figure(-self.s, self.s, -self.s, self.s)
            table = init_table(rules, self.legend, fig_leg, ax_leg)
            if self.record_data:
                simulation.run_with_data(
                    self.record_data,
                    self.num_steps,
                    mmsi_list,
                    rules,
                    table,
                    ax,
                    self.Ɛ,
                    self.s,
                )
            else:
                simulation.run(
                    self.record_data,
                    self.num_steps,
                    mmsi_list,
                    rules,
                    table,
                    ax,
                    ax_leg,
                    self.Ɛ,
                    self.s,
                )
        else:
            if self.record_data:
                simulation.run_with_data(
                    self.record_data,
                    self.num_steps,
                    mmsi_list,
                    rules,
                    None,
                    None,
                    self.Ɛ,
                    self.s,
                )
            else:
                simulation.run(
                    self.record_data,
                    self.num_steps,
                    mmsi_list,
                    rules,
                    None,
                    None,
                    None,
                    self.Ɛ,
                    self.s,
                )
        return sea_objects
