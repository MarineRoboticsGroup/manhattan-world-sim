import os
from os.path import join, expanduser
import sys
sys.path.insert(0, os.path.abspath(".."))

from manhattan.simulator.simulator import ManhattanSimulator, SimulationParams

import numpy as np
np.random.seed(999)

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

num_timesteps = 100
for _ in range(num_timesteps):
    sim.random_step()
    # sim.plot_current_state(show_grid=True)

data_dir = expanduser(
    join("~", "data", "example_factor_graphs")
)
sim.save_simulation_data(data_dir, format='chad')