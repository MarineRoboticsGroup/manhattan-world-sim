import os
from os.path import join, expanduser
import sys
import flamegraph
import subprocess
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(".."))

from manhattan.simulator.simulator import ManhattanSimulator, SimulationParams

import numpy as np

np.random.seed(999)

cwd = os.getcwd()
fg_log_path = f"{cwd}/flamegraph.log"
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
    debug_mode=False,
)
sim = ManhattanSimulator(sim_args)

num_robots = 1
num_beacons = 3
sim.add_robots(num_robots)
sim.add_beacons(num_beacons)

num_timesteps = 10000
for _ in tqdm(range(num_timesteps)):
    sim.random_step()

fg_thread.stop()
fg_image_path = f"{cwd}/flamegraph.svg"
fg_script_path = f"{cwd}/FlameGraph/flamegraph.pl"
fg_bash_command = (
    f"bash {cwd}/flamegraph.bash {fg_script_path} {fg_log_path} {fg_image_path}"
)
subprocess.call(fg_bash_command.split(), stdout=subprocess.PIPE)
