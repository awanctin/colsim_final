import os
import random

import numpy as np
import plotly.graph_objects as go

from ..experiment_constants import R_VESSEL_M

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "output", "3D"
)

COLOURS = ("red", "blue", "green", "purple", "orange", "pink", "brown", "black", "grey")
_COLOUR_RNG = random.Random(0)


def pick_colour():
    """Pick a random colour from the list.

    Returns:
        str: Colour name
    """
    return _COLOUR_RNG.choice(COLOURS)


class Representation:
    def __init__(self, radius_x=R_VESSEL_M, radius_y=R_VESSEL_M, num_circle_points=20):
        """Set the tube size and empty the plot data.

        Args:
            radius_x (float): Tube radius along x, m
            radius_y (float): Tube radius along y, m
            num_circle_points (int): Points per tube section
        """
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.num_circle_points = num_circle_points
        self.tube_x = []
        self.tube_y = []
        self.tube_z = []
        self.faces = []
        self.traces = []

    def find_overlap(self, x1, y1, z1, x2, y2, z2):
        """Mark the points where two trajectories come closer than the radius.

        Args:
            x1 (list): First x values
            y1 (list): First y values
            z1 (list): First times
            x2 (list): Second x values
            y2 (list): Second y values
            z2 (list): Second times
        """
        p1 = np.column_stack([x1, y1, z1]).astype(float)
        p2 = np.column_stack([x2, y2, z2]).astype(float)
        radius = max(self.radius_x, self.radius_y)
        d = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=-1)
        i, j = np.nonzero(d < radius)
        if len(i) > 1:
            mid = 0.5 * (p1[i] + p2[j])
            self.traces.append(
                go.Scatter3d(
                    x=mid[:, 0],
                    y=mid[:, 1],
                    z=mid[:, 2],
                    mode="markers",
                    marker=dict(size=2, color="green"),
                    opacity=0.3,
                    name="overlap",
                )
            )

    def draw_tube(self, x, y, z, headings):
        """Build a tube around a trajectory, turned with the heading.

        Args:
            x (list): x values, m
            y (list): y values, m
            z (list): Times
            headings (list): Headings, rad

        Returns:
            tuple: Tube x, y and z points
        """
        tube_x = []
        tube_y = []
        tube_z = []

        for i in range(len(x)):
            theta = np.linspace(0, 2 * np.pi, self.num_circle_points)
            ellipse_x = self.radius_x * np.cos(theta)
            ellipse_y = self.radius_y * np.sin(theta)

            heading_vector = np.array([np.cos(headings[i]), np.sin(headings[i])])
            rotation_matrix = np.array(
                [
                    [heading_vector[0], -heading_vector[1]],
                    [heading_vector[1], heading_vector[0]],
                ]
            )

            rotated_points = np.dot(rotation_matrix, np.array([ellipse_x, ellipse_y]))

            for j in range(self.num_circle_points):
                point_x = x[i] + rotated_points[0, j]
                point_y = y[i] + rotated_points[1, j]
                point_z = z[i]

                tube_x.append(point_x)
                tube_y.append(point_y)
                tube_z.append(point_z)

        return tube_x, tube_y, tube_z

    def create_3D_rep(self, all_sea_objects):
        """Build the 3D traces of every trajectory over time.

        Args:
            all_sea_objects (dict): Recorded states per vessel
        """
        self.tube_x = []
        self.tube_y = []
        self.tube_z = []
        self.faces = []
        self.traces = []

        offset = 0
        keys = list(all_sea_objects.keys())

        for i in range(len(keys)):
            key = keys[i]
            value = all_sea_objects[key]
            x = [v[1][0][0] for v in value]
            y = [v[1][1][0] for v in value]
            theta = [v[1][3][0] for v in value]
            t = np.linspace(0, len(x), len(x))

            tube_x, tube_y, tube_z = self.draw_tube(x, y, t, theta)

            num_points = len(tube_x)
            self.tube_x += tube_x
            self.tube_y += tube_y
            self.tube_z += tube_z

            for j in range(len(x) - 1):
                for k in range(self.num_circle_points):
                    next_k = (k + 1) % self.num_circle_points
                    self.faces.append(
                        [
                            offset + j * self.num_circle_points + k,
                            offset + (j + 1) * self.num_circle_points + k,
                            offset + (j + 1) * self.num_circle_points + next_k,
                        ]
                    )
                    self.faces.append(
                        [
                            offset + j * self.num_circle_points + k,
                            offset + (j + 1) * self.num_circle_points + next_k,
                            offset + j * self.num_circle_points + next_k,
                        ]
                    )

            offset += num_points

            color_choice = pick_colour()
            self.traces.append(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=t,
                    mode="lines",
                    line=dict(color=color_choice),
                    name=f"Trajectory {key}",
                )
            )
            self.traces.append(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=[0] * len(t),
                    mode="lines",
                    line=dict(dash="dot", color=color_choice),
                    name=f"Projection {key}",
                )
            )

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                key1 = keys[i]
                key2 = keys[j]
                value1 = all_sea_objects[key1]
                value2 = all_sea_objects[key2]
                x1 = [v[1][0][0] for v in value1]
                y1 = [v[1][1][0] for v in value1]
                theta1 = [v[1][3][0] for v in value1]
                t1 = np.linspace(0, len(x1), len(x1))

                x2 = [v[1][0][0] for v in value2]
                y2 = [v[1][1][0] for v in value2]
                theta2 = [v[1][3][0] for v in value2]
                t2 = np.linspace(0, len(x2), len(x2))

                tube_x1, tube_y1, tube_z1 = self.draw_tube(x1, y1, t1, theta1)
                tube_x2, tube_y2, tube_z2 = self.draw_tube(x2, y2, t2, theta2)

                self.find_overlap(tube_x1, tube_y1, tube_z1, tube_x2, tube_y2, tube_z2)

        if self.faces:
            i, j, k = np.array(self.faces).T

            self.traces.append(
                go.Mesh3d(
                    x=self.tube_x,
                    y=self.tube_y,
                    z=self.tube_z,
                    i=i,
                    j=j,
                    k=k,
                    opacity=0.5,
                    color="lightblue",
                )
            )

    def show(self):
        """Show the 3D figure in a browser."""
        fig = go.Figure(
            data=[trace for trace in self.traces if isinstance(trace, go.Scatter3d)]
            + [trace for trace in self.traces if isinstance(trace, go.Mesh3d)],
            layout=go.Layout(
                title="3D Representation of Boat Trajectories",
                scene=dict(
                    xaxis_title="X Axis",
                    yaxis_title="Y Axis",
                    zaxis_title="Time",
                    aspectmode="data",
                ),
            ),
        )
        fig.show()

    def save(
        self,
        name="traj_overlap.html",
        path=OUTPUT_DIR,
    ):
        """Save the 3D figure as an HTML file.

        Args:
            name (str): File name
            path (str): Output folder
        """
        fig = go.Figure(
            data=[trace for trace in self.traces if isinstance(trace, go.Scatter3d)]
            + [trace for trace in self.traces if isinstance(trace, go.Mesh3d)],
            layout=go.Layout(
                title="3D Representation of Boat Trajectories",
                scene=dict(
                    xaxis_title="X Axis",
                    yaxis_title="Y Axis",
                    zaxis_title="Time",
                    aspectmode="data",
                ),
            ),
        )
        os.makedirs(path, exist_ok=True)
        final_path = os.path.join(path, name)
        fig.write_html(final_path)


if __name__ == "__main__":
    all_sea_objects = {
        "539_Boat": [
            [
                "Boat",
                [[-0.85840028], [10.7792047], [1.34171896], [4.26807024]],
                "0",
                {},
            ],
            ["Boat", [[-0.91607312], [10.6580604], [1.34171896], [4.271119]], "0", {}],
            [
                "Boat",
                [[-0.97337634], [10.53674083], [1.34171896], [4.27266484]],
                "0",
                {},
            ],
            [
                "Boat",
                [[-1.03049196], [10.41533282], [1.34171896], [4.27344871]],
                "0",
                {},
            ],
            [
                "Boat",
                [[-1.08751239], [10.29388008], [1.34171896], [4.27384624]],
                "0",
                {},
            ],
            [
                "Boat",
                [[-1.14448453], [10.17240468], [1.34171896], [4.27404787]],
                "0",
                {},
            ],
            [
                "Boat",
                [[-1.20143219], [10.0509178], [1.34171896], [4.01285419]],
                "1",
                {
                    619: [
                        [
                            [
                                [[3.25620036], [7.83770397], [2.46697896], [2.973593]],
                                [
                                    [3.01297567],
                                    [7.87895445],
                                    [2.46697896],
                                    [2.96612544],
                                ],
                                [[2.7700658], [7.92202006], [2.46697896], [2.96229519]],
                                [
                                    [2.52732267],
                                    [7.96601575],
                                    [2.46697896],
                                    [2.96032994],
                                ],
                                [
                                    [2.28466646],
                                    [8.01048841],
                                    [2.46697896],
                                    [2.95932126],
                                ],
                                [
                                    [2.04205524],
                                    [8.05520581],
                                    [2.46697896],
                                    [2.95880336],
                                ],
                            ]
                        ]
                    ]
                },
            ],
            [
                "Boat",
                [[-1.28782034], [9.94825728], [1.34171896], [4.22541554]],
                "1",
                {
                    619: [
                        [
                            [
                                [[3.25620036], [7.83770397], [2.46697896], [2.973593]],
                                [
                                    [3.01297567],
                                    [7.87895445],
                                    [2.46697896],
                                    [2.96612544],
                                ],
                                [[2.7700658], [7.92202006], [2.46697896], [2.96229519]],
                                [
                                    [2.52732267],
                                    [7.96601575],
                                    [2.46697896],
                                    [2.96032994],
                                ],
                                [
                                    [2.28466646],
                                    [8.01048841],
                                    [2.46697896],
                                    [2.95932126],
                                ],
                                [
                                    [2.04205524],
                                    [8.05520581],
                                    [2.46697896],
                                    [2.95880336],
                                ],
                                [[1.7994672], [8.10004885], [2.46697896], [2.98730586]],
                            ]
                        ]
                    ]
                },
            ],
            [
                "Boat",
                [[-1.35060653], [9.82968243], [1.34171896], [4.43040621]],
                "1",
                {
                    619: [
                        [
                            [
                                [[3.25620036], [7.83770397], [2.46697896], [2.973593]],
                                [
                                    [3.01297567],
                                    [7.87895445],
                                    [2.46697896],
                                    [2.96612544],
                                ],
                                [[2.7700658], [7.92202006], [2.46697896], [2.96229519]],
                                [
                                    [2.52732267],
                                    [7.96601575],
                                    [2.46697896],
                                    [2.96032994],
                                ],
                                [
                                    [2.28466646],
                                    [8.01048841],
                                    [2.46697896],
                                    [2.95932126],
                                ],
                                [
                                    [2.04205524],
                                    [8.05520581],
                                    [2.46697896],
                                    [2.95880336],
                                ],
                                [[1.7994672], [8.10004885], [2.46697896], [2.98730586]],
                                [
                                    [1.55569974],
                                    [8.13796025],
                                    [2.46697896],
                                    [3.26935341],
                                ],
                            ]
                        ]
                    ]
                },
            ],
            [
                "Boat",
                [[-1.38794129], [9.70080958], [1.34171896], [4.63354732]],
                "1",
                {
                    619: [
                        [
                            [
                                [[3.25620036], [7.83770397], [2.46697896], [2.973593]],
                                [
                                    [3.01297567],
                                    [7.87895445],
                                    [2.46697896],
                                    [2.96612544],
                                ],
                                [[2.7700658], [7.92202006], [2.46697896], [2.96229519]],
                                [
                                    [2.52732267],
                                    [7.96601575],
                                    [2.46697896],
                                    [2.96032994],
                                ],
                                [
                                    [2.28466646],
                                    [8.01048841],
                                    [2.46697896],
                                    [2.95932126],
                                ],
                                [
                                    [2.04205524],
                                    [8.05520581],
                                    [2.46697896],
                                    [2.95880336],
                                ],
                                [[1.7994672], [8.10004885], [2.46697896], [2.98730586]],
                                [
                                    [1.55569974],
                                    [8.13796025],
                                    [2.46697896],
                                    [3.26935341],
                                ],
                                [
                                    [1.31101251],
                                    [8.10652762],
                                    [2.46697896],
                                    [3.52789666],
                                ],
                            ]
                        ]
                    ]
                },
            ],
        ],
        "619_Boat": [
            ["Boat", [[3.25620036], [7.83770397], [2.46697896], [2.973593]], "0", {}],
            ["Boat", [[3.01297567], [7.87895445], [2.46697896], [2.96612544]], "0", {}],
            ["Boat", [[2.7700658], [7.92202006], [2.46697896], [2.96229519]], "0", {}],
            ["Boat", [[2.52732267], [7.96601575], [2.46697896], [2.96032994]], "0", {}],
            ["Boat", [[2.28466646], [8.01048841], [2.46697896], [2.95932126]], "0", {}],
            ["Boat", [[2.04205524], [8.05520581], [2.46697896], [2.95880336]], "0", {}],
            [
                "Boat",
                [[1.7994672], [8.10004885], [2.46697896], [2.98730586]],
                "1",
                {
                    539: [
                        [
                            [
                                [
                                    [-0.85840028],
                                    [10.7792047],
                                    [1.34171896],
                                    [4.26807024],
                                ],
                                [[-0.91607312], [10.6580604], [1.34171896], [4.271119]],
                                [
                                    [-0.97337634],
                                    [10.53674083],
                                    [1.34171896],
                                    [4.27266484],
                                ],
                                [
                                    [-1.03049196],
                                    [10.41533282],
                                    [1.34171896],
                                    [4.27344871],
                                ],
                                [
                                    [-1.08751239],
                                    [10.29388008],
                                    [1.34171896],
                                    [4.27384624],
                                ],
                                [
                                    [-1.14448453],
                                    [10.17240468],
                                    [1.34171896],
                                    [4.27404787],
                                ],
                                [
                                    [-1.20143219],
                                    [10.0509178],
                                    [1.34171896],
                                    [4.01285419],
                                ],
                            ]
                        ]
                    ]
                },
            ],
            [
                "Boat",
                [[1.55569974], [8.13796025], [2.46697896], [3.26935341]],
                "1",
                {
                    539: [
                        [
                            [
                                [
                                    [-0.85840028],
                                    [10.7792047],
                                    [1.34171896],
                                    [4.26807024],
                                ],
                                [[-0.91607312], [10.6580604], [1.34171896], [4.271119]],
                                [
                                    [-0.97337634],
                                    [10.53674083],
                                    [1.34171896],
                                    [4.27266484],
                                ],
                                [
                                    [-1.03049196],
                                    [10.41533282],
                                    [1.34171896],
                                    [4.27344871],
                                ],
                                [
                                    [-1.08751239],
                                    [10.29388008],
                                    [1.34171896],
                                    [4.27384624],
                                ],
                                [
                                    [-1.14448453],
                                    [10.17240468],
                                    [1.34171896],
                                    [4.27404787],
                                ],
                                [
                                    [-1.20143219],
                                    [10.0509178],
                                    [1.34171896],
                                    [4.01285419],
                                ],
                                [
                                    [-1.28782034],
                                    [9.94825728],
                                    [1.34171896],
                                    [4.22541554],
                                ],
                            ]
                        ]
                    ]
                },
            ],
            [
                "Boat",
                [[1.31101251], [8.10652762], [2.46697896], [3.52789666]],
                "1",
                {
                    539: [
                        [
                            [
                                [
                                    [-0.85840028],
                                    [10.7792047],
                                    [1.34171896],
                                    [4.26807024],
                                ],
                                [[-0.91607312], [10.6580604], [1.34171896], [4.271119]],
                                [
                                    [-0.97337634],
                                    [10.53674083],
                                    [1.34171896],
                                    [4.27266484],
                                ],
                                [
                                    [-1.03049196],
                                    [10.41533282],
                                    [1.34171896],
                                    [4.27344871],
                                ],
                                [
                                    [-1.08751239],
                                    [10.29388008],
                                    [1.34171896],
                                    [4.27384624],
                                ],
                                [
                                    [-1.14448453],
                                    [10.17240468],
                                    [1.34171896],
                                    [4.27404787],
                                ],
                                [
                                    [-1.20143219],
                                    [10.0509178],
                                    [1.34171896],
                                    [4.01285419],
                                ],
                                [
                                    [-1.28782034],
                                    [9.94825728],
                                    [1.34171896],
                                    [4.22541554],
                                ],
                                [
                                    [-1.35060653],
                                    [9.82968243],
                                    [1.34171896],
                                    [4.43040621],
                                ],
                            ]
                        ]
                    ]
                },
            ],
            [
                "Boat",
                [[1.08249429], [8.0135799], [2.46697896], [3.7607972]],
                "1",
                {
                    539: [
                        [
                            [
                                [
                                    [-0.85840028],
                                    [10.7792047],
                                    [1.34171896],
                                    [4.26807024],
                                ],
                                [[-0.91607312], [10.6580604], [1.34171896], [4.271119]],
                                [
                                    [-0.97337634],
                                    [10.53674083],
                                    [1.34171896],
                                    [4.27266484],
                                ],
                                [
                                    [-1.03049196],
                                    [10.41533282],
                                    [1.34171896],
                                    [4.27344871],
                                ],
                                [
                                    [-1.08751239],
                                    [10.29388008],
                                    [1.34171896],
                                    [4.27384624],
                                ],
                                [
                                    [-1.14448453],
                                    [10.17240468],
                                    [1.34171896],
                                    [4.27404787],
                                ],
                                [
                                    [-1.20143219],
                                    [10.0509178],
                                    [1.34171896],
                                    [4.01285419],
                                ],
                                [
                                    [-1.28782034],
                                    [9.94825728],
                                    [1.34171896],
                                    [4.22541554],
                                ],
                                [
                                    [-1.35060653],
                                    [9.82968243],
                                    [1.34171896],
                                    [4.43040621],
                                ],
                                [
                                    [-1.38794129],
                                    [9.70080958],
                                    [1.34171896],
                                    [4.63354732],
                                ],
                            ]
                        ]
                    ]
                },
            ],
        ],
    }

    rep = Representation()
    rep.create_3D_rep(all_sea_objects)
    rep.show()
