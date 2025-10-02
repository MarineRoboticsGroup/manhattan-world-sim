import os
from os.path import join, expanduser
import sys

sys.path.insert(0, os.path.abspath(".."))

from manhattan.simulator.simulator import ManhattanSimulator, SimulationParams

show_animation = False
num_beacons = 0
num_robots = 2
grid_len = 20
range_prob = 1.0
dist_stddev = 1.0
pos_stddev = 0.5
theta_stddev = 0.01
seed_cnt = 10
num_timesteps = 2

sim_args = SimulationParams(
    num_robots=num_robots,
    num_beacons=num_beacons,
    grid_shape=(grid_len, grid_len),
    y_steps_to_intersection=2,
    x_steps_to_intersection=2,
    cell_scale=1.0,
    range_sensing_prob=range_prob,
    range_sensing_radius=100.0,
    false_range_data_association_prob=0.0,
    outlier_prob=0.0,
    max_num_loop_closures=000,
    loop_closure_prob=1.0,
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

from py_factor_graph.io.pyfg_text import save_to_pyfg_text

# fpath = expanduser(join("~", "test.pyfg"))
fpath = expanduser(join("~", "data_matrix_1_landmarks_wo_loop_closures.pyfg"))
fpath = expanduser(join("~", "two_timesteps_robot_only.pyfg"))
# sim._factor_graph.print_summary()

from py_factor_graph.modifiers import convert_to_sensor_network_localization, add_random_range_measurements, make_all_ranges_perfect, make_fully_connected_ranges_between_all_landmarks

pyfg = sim._factor_graph
# pyfg = convert_to_sensor_network_localization(pyfg)
# pyfg = add_random_range_measurements(pyfg, num_measures=0, stddev=1.0, prob_from_pose_variable=0.0, prob_to_pose_variable=0.0)
# pyfg = make_fully_connected_ranges_between_all_landmarks(pyfg)
pyfg = make_all_ranges_perfect(pyfg)
save_to_pyfg_text(pyfg, fpath)
pyfg.print_summary()
# save_to_pyfg_text(sim._factor_graph, fpath)
print(f"Saved data to {fpath}")

# sim._factor_graph.animate_odometry(show_gt=True)
true_traj = sim._factor_graph.true_trajectories[0]
# for i in range(5):
    # print(true_traj[i])

# print(sim._factor_graph.landmark_variables[0])
