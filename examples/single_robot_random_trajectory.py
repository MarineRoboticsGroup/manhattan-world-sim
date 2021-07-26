import os
import sys
sys.path.insert(0, os.path.abspath(".."))

from manhattan.simulator.simulator import ManhattanSimulator, SimulationParams

sim_args = SimulationParams(
    grid_shape=(20, 20),
    row_corner_number=2,
    column_corner_number=2,
    cell_scale=1.0,
    range_sensing_prob=0.5,
    range_sensing_radius=10.0,
    false_range_data_association_prob=0.1,
    outlier_prob=0.1,
    loop_closure_prob=0.1,
    loop_closure_radius=3.0,
    false_loop_closure_prob=0.1,
)
sim = ManhattanSimulator(sim_args)

num_robots = 3
num_beacons = 2
sim.add_robots(num_robots)
sim.add_beacons(num_beacons)

num_timesteps = 100
for _ in range(num_timesteps):
    sim.random_step()
    # sim.move_robots_randomly()
    sim.plot_current_state(show_grid=True)
