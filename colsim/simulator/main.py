import matplotlib
import matplotlib.pyplot as plt
from .simulation_runner import SimulationRunner
from tqdm.auto import tqdm
import time
import random

IMAZU_HALF_EXTENT_M = 11_000.0

if __name__ == "__main__":
    nb_simulations = 100
    matplotlib.rcParams["interactive"] = False
    start_time = time.time()
    trajs = []

    for i in tqdm(range(nb_simulations), desc="Simulations"):
        runner = SimulationRunner()
        runner.s = IMAZU_HALF_EXTENT_M
        runner.num_steps = 200
        runner.record_data = True
        runner.visu_figure = False
        runner.render_3D = False
        runner.imazu = True
        runner.imazu_number = 2
        sea_objects = runner.run()

        for sea_object in sea_objects:
            trajs.append(sea_object.trajectory)

    with open(f"simulation_trajectories_0.txt", "w") as f:
        for traj in trajs:
            f.write(f"{traj[0]}\n{traj[1]}\n\n")

    fig, ax = plt.subplots()
    min_x = min([min(traj[0]) for traj in trajs])
    max_x = max([max(traj[0]) for traj in trajs])
    min_y = min([min(traj[1]) for traj in trajs])
    max_y = max([max(traj[1]) for traj in trajs])
    for traj in trajs:
        x = traj[0]
        y = traj[1]
        ax.plot(x, y, marker="o", markersize=2, linewidth=1)
    plt.title("Simulation Trajectories")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.xlim(min_x - 1, max_x + 1)
    plt.ylim(min_y - 1, max_y + 1)
    plt.grid()
    plt.savefig("simulation_trajectories_0.png", dpi=600)
    plt.show()
