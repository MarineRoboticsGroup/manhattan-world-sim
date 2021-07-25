import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from manhattan.simulator.simulator import ManhattanSimulator

# from manhattan.environment.environment import ManhattanWorld


sim_args = ManhattanSimulator.SimulationParams(
    grid_shape=(20, 20),
    row_corner_number=2,
    column_corner_number=2,
    cell_scale=1.0,
    range_sensing_prob=0.5,
    ambiguous_data_association_prob=0.1,
    outlier_prob=0.1,
    loop_closure_prob=0.1,
    loop_closure_radius=3.0,
)
sim = ManhattanSimulator(sim_args)

num_robots = 3
num_beacons = 2
sim.add_robots(num_robots)
sim.add_beacons(num_beacons)

num_timesteps = 10
for _ in range(num_timesteps):
    sim.move_robots_randomly()

sim.print_simulator_state()
sim.plot_current_state(show_grid=True)
