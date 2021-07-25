from src.simulator.simulator import ManhattanSimulator
from src.environment.environment import ManhattanWorld


sim_args = ManhattanSimulator.simulator_args(range_sensing_prob=0.5,
            row_corner_number=3,
            column_corner_number=4,
            ambiguous_data_association_prob=0.1,
            outlier_prob=0.1,
            loop_closure_prob=0.1,
            loop_closure_radius=3,)
sim = ManhattanSimulator(sim_args)
print(sim)