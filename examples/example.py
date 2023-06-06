import os
from os.path import join, expanduser
import sys

sys.path.insert(0, os.path.abspath(".."))

from manhattan.simulator.simulator import ManhattanSimulator, SimulationParams

show_animation = False
num_beacons = 1
grid_len = 30
range_prob = 0.25
dist_stddev = 0.1
pos_stddev = 0.1
theta_stddev = 0.1
seed_cnt = 0
num_timesteps = 100

sim_args = SimulationParams(
    num_robots=1,
    num_beacons=num_beacons,
    grid_shape=(grid_len, grid_len),
    y_steps_to_intersection=2,
    x_steps_to_intersection=5,
    cell_scale=1.0,
    range_sensing_prob=range_prob,
    range_sensing_radius=100.0,
    false_range_data_association_prob=0.0,
    outlier_prob=0.0,
    max_num_loop_closures=0,
    loop_closure_prob=0.05,
    loop_closure_radius=20.0,
    false_loop_closure_prob=0.0,
    range_stddev=dist_stddev,
    odom_x_stddev=pos_stddev,
    odom_y_stddev=pos_stddev,
    odom_theta_stddev=theta_stddev,
    loop_x_stddev=pos_stddev,
    loop_y_stddev=pos_stddev,
    loop_theta_stddev=theta_stddev,
    debug_mode=False,
    seed_num=(seed_cnt + 1) * 9999,
    groundtruth_measurements=True,
    # no_loop_pose_idx=[0, 1, 2],
    # exclude_last_n_poses_for_loop_closure=2
)
sim = ManhattanSimulator(sim_args)

if show_animation:
    sim.plot_grid()
    sim.plot_beacons()

for _ in range(num_timesteps):
    sim.random_step()

    if show_animation:
        sim.plot_robot_states()
        sim.show_plot(animation=True)

if show_animation:
    sim.close_plot()

data_dir = expanduser(
    join(
        "~",
        "data",
        "manhattan",
        f"{num_timesteps}_timesteps",
        f"{int(pos_stddev*1000)}_pos_stddev",
        f"{int(theta_stddev*1000)}_rot_stddev",
        f"{int(range_prob*1000)}_range_prob",
        f"{num_beacons}_beacons",
        f"{grid_len}_grid",
        f"seed_{sim_args.seed_num}",
    )
)
sim.save_simulation_data(data_dir, format="pickle")
sim._factor_graph.print_summary()
print(f"Saved data to {data_dir}")
