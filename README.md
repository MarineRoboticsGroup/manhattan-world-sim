# Manhattan-waterworld

This is a repository for generating random Manhattan-World-style experiments
with ambiguous data association, multiple robots, and range measurements between
both robots and static beacons placed in the environment.

## Getting Started

To run some sample code:

``` Bash
# setup the environment (name assigned from the .yml file)
conda env create -f env.yml
conda activate manhattan_sim

# run the example code from the example directory of this repository
cd examples
python single_robot_random_trajectory.py
```
