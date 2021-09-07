import os
from os.path import join, expanduser
import sys

sys.path.insert(0, os.path.abspath(".."))

from manhattan.simulator.simulator import ManhattanSimulator, SimulationParams

import numpy as np

for j in range(1, 3):
    print(f"\nSimulating {j} loop closures")
    for i in range(1, 11):
        sim_args = SimulationParams(
            num_robots=1,
            num_beacons=0,
            grid_shape=(2, 2),
            y_steps_to_intersection=1,
            x_steps_to_intersection=1,
            cell_scale=20.0,
            range_sensing_prob=0.5,
            range_sensing_radius=40.0,
            false_range_data_association_prob=0.0,
            outlier_prob=0.1,
            max_num_loop_closures=j,
            loop_closure_prob=0.9,
            loop_closure_radius=99999.0,
            false_loop_closure_prob=1.0,
            range_stddev=1e-1,
            odom_x_stddev=1e0,
            odom_y_stddev=0.2,
            odom_theta_stddev=1e-1,
            loop_x_stddev=1e0,
            loop_y_stddev=0.2,
            loop_theta_stddev=1e-1,
            debug_mode=False,
            seed_num=i,
            groundtruth_measurements=True,
        )
        sim = ManhattanSimulator(sim_args)

        num_timesteps = 4
        # sim.plot_robot_states()
        # sim.show_plot(animation=True)
        for _ in range(num_timesteps):
            sim.random_step()
            # sim.plot_robot_states()
            # sim.show_plot(animation=True)

        data_dir = expanduser(
            join(
                "~",
                "data",
                "example_factor_graphs",
                f"{sim_args.max_num_loop_closures}_loop_clos",
                f"test_{sim_args.seed_num}",
            )
        )
        sim.save_simulation_data(data_dir, format="efg")
