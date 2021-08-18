import os
from os.path import join, expanduser
import sys
import cProfile, pstats, io
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(".."))

from manhattan.simulator.simulator import ManhattanSimulator, SimulationParams

import numpy as np

np.random.seed(999)

pr = cProfile.Profile()
pr.enable()

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

pr.disable()
s = io.StringIO()
sort_by = 'cumtime'
ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
ps.print_stats(.1)
print(s.getvalue())