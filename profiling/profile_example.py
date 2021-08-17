import os
from os.path import join, expanduser
import sys
import flamegraph
import subprocess

sys.path.insert(0, os.path.abspath(".."))

from manhattan.simulator.simulator import ManhattanSimulator, SimulationParams

import numpy as np

np.random.seed(999)

cwd = os.getcwd()
fg_log_path = f"{cwd}/profile_code.log"
fg_thread = flamegraph.start_profile_thread(fd=open(fg_log_path, "w"))

sim_args = SimulationParams(
    grid_shape=(20, 20),
    row_corner_number=2,
    column_corner_number=2,
    cell_scale=1.0,
    range_sensing_prob=0.5,
    range_sensing_radius=40.0,
    false_range_data_association_prob=0.0,
    outlier_prob=0.1,
    loop_closure_prob=0.1,
    loop_closure_radius=3.0,
    false_loop_closure_prob=0.1,
)
sim = ManhattanSimulator(sim_args)

num_robots = 1
num_beacons = 3
sim.add_robots(num_robots)
sim.add_beacons(num_beacons)

num_timesteps = 999
for _ in range(num_timesteps):
    sim.random_step()
    # sim.plot_current_state(show_grid=True)

data_dir = expanduser(join("~", "data", "example_factor_graphs"))
sim.save_simulation_data(data_dir, format="chad")

fg_thread.stop()
fg_image_path = f"{cwd}/profile.svg"
fg_script_path = f"{cwd}/FlameGraph/flamegraph.pl"
fg_bash_command = f"bash {cwd}/flamegraph.bash {fg_script_path} {fg_log_path} {fg_image_path}"
subprocess.call(fg_bash_command.split(), stdout=subprocess.PIPE)

