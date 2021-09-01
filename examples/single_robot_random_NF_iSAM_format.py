import os
from os.path import join, expanduser
import sys

sys.path.insert(0, os.path.abspath(".."))

from manhattan.simulator.simulator import ManhattanSimulator, SimulationParams

import numpy as np

np.random.seed(999)

sim_args = SimulationParams(
    grid_shape=(30, 30),
    y_steps_to_intersection=5,
    x_steps_to_intersection=5,
    cell_scale=1.0,
    range_sensing_prob=0.5,
    range_sensing_radius=40.0,
    false_range_data_association_prob=0.0,
    outlier_prob=0.1,
    loop_closure_prob=0.1,
    loop_closure_radius=3.0,
    false_loop_closure_prob=0.1,
    range_stddev=1e-1,
    odom_x_stddev=1e-1,
    odom_y_stddev=1e-1,
    odom_theta_stddev=1e-2,
    debug_mode=False,
    groundtruth_measurements=False,
)
sim = ManhattanSimulator(sim_args)


num_robots = 2
sim.add_robots(num_robots)

num_beacons = 1
sim.add_beacons(num_beacons)

num_timesteps = 5
for _ in range(num_timesteps):
    sim.random_step()
    sim.plot_robot_states()
    sim.show_plot(animation=True)

data_dir = "/home/alan/"
data_dir = expanduser(join("~", "data", "example_factor_graphs"))
sim.save_simulation_data(data_dir, format="efg")
