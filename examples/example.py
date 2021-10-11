import os
from os.path import join, expanduser
import sys

sys.path.insert(0, os.path.abspath(".."))

from manhattan.simulator.simulator import ManhattanSimulator, SimulationParams

import numpy as np

for num_timesteps in [10, 100, 1000, 10000, 100000]:
    for pos_stddev in [1e-2, 1e-1]:
        for theta_stddev in [1e-3, 1e-2, 1e-1]:
            for dist_stddev in [1e-2, 1e-1, 1]:
                for num_beacons in [1, 3, 5]:
                    for grid_len in [10, 20, 50]:
                        print()
                        for i in range(0, 10):
                            sim_args = SimulationParams(
                                num_robots=1,
                                num_beacons=3,
                                grid_shape=(10, 10),
                                y_steps_to_intersection=2,
                                x_steps_to_intersection=2,
                                cell_scale=1.0,
                                range_sensing_prob=0.8,
                                range_sensing_radius=999.0,
                                false_range_data_association_prob=0.0,
                                outlier_prob=0.0,
                                max_num_loop_closures=0,
                                loop_closure_prob=0.0,
                                loop_closure_radius=9999.0,
                                false_loop_closure_prob=0.0,
                                range_stddev=1e-1,
                                odom_x_stddev=1e-1,
                                odom_y_stddev=1e-1,
                                odom_theta_stddev=1e-3,
                                loop_x_stddev=1e-2,
                                loop_y_stddev=1e-2,
                                loop_theta_stddev=1e-3,
                                debug_mode=False,
                                seed_num=i,
                                groundtruth_measurements=True,
                                # no_loop_pose_idx=[0, 1, 2],
                                # exclude_last_n_poses_for_loop_closure=2
                            )
                            sim = ManhattanSimulator(sim_args)

                            for _ in range(num_timesteps):
                                sim.random_step()
                            sim.close_plot()

                            data_dir = expanduser(
                                join(
                                    "~",
                                    "data",
                                    "example_factor_graphs",
                                    f"{num_timesteps}_timesteps",
                                    f"{int(pos_stddev*1000)}_pos_stddev",
                                    f"{int(theta_stddev*1000)}_rot_stddev",
                                    f"{num_beacons}_beacons",
                                    f"{grid_len}_grid",
                                    f"seed_{sim_args.seed_num}",
                                )
                            )
                            sim.save_simulation_data(data_dir, format="efg")
