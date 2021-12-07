import os
from os.path import join, expanduser
import sys

sys.path.insert(0, os.path.abspath(".."))

from manhattan.simulator.simulator import ManhattanSimulator, SimulationParams

import numpy as np

for num_timesteps in [5000]:
    for pos_stddev in [1e-1]:
        for theta_stddev in [1e-2]:
            for dist_stddev in [1e-3]:
                for num_beacons in [1]:
                    for loop_prob in [0.05]:
                        for grid_len in [10]:
                            print()
                            for i in range(0, 1):
                                sim_args = SimulationParams(
                                    num_robots=3,
                                    num_beacons=num_beacons,
                                    grid_shape=(grid_len, grid_len),
                                    y_steps_to_intersection=2,
                                    x_steps_to_intersection=2,
                                    cell_scale=1.0,
                                    range_sensing_prob=0.5,
                                    range_sensing_radius=999.0,
                                    false_range_data_association_prob=0.0,
                                    outlier_prob=0.0,
                                    max_num_loop_closures=9999999,
                                    loop_closure_prob=loop_prob,
                                    loop_closure_radius=10.0,
                                    false_loop_closure_prob=0.0,
                                    range_stddev=1e-1,
                                    odom_x_stddev=pos_stddev,
                                    odom_y_stddev=pos_stddev,
                                    odom_theta_stddev=theta_stddev,
                                    loop_x_stddev=pos_stddev,
                                    loop_y_stddev=pos_stddev,
                                    loop_theta_stddev=theta_stddev,
                                    debug_mode=False,
                                    seed_num=i,
                                    groundtruth_measurements=False,
                                    # no_loop_pose_idx=[0, 1, 2],
                                    # exclude_last_n_poses_for_loop_closure=2
                                )
                                sim = ManhattanSimulator(sim_args)

                                sim.plot_grid()
                                sim.plot_beacons()

                                for _ in range(num_timesteps):
                                    sim.random_step()
                                    sim.plot_robot_states()
                                    sim.show_plot(animation=True)
                                sim.close_plot()

                                data_dir = expanduser(
                                    join(
                                        "~",
                                        "data",
                                        "manhattan",
                                        f"{num_timesteps}_timesteps",
                                        f"{int(pos_stddev*1000)}_pos_stddev",
                                        f"{int(theta_stddev*1000)}_rot_stddev",
                                        f"{int(loop_prob*100)}_loop_prob",
                                        f"{num_beacons}_beacons",
                                        f"{grid_len}_grid",
                                        f"seed_{sim_args.seed_num}",
                                    )
                                )
                                sim.save_simulation_data(data_dir, format="pickle")
