from Environment import *

class ManhattanSimulator:
    """This class defines a simulator using Manhattan world-like environments
    :param
    env: simulated environment
    """
    def __init__(self, env: ManhattanWaterworld, robot_step_length: float):
        assert robot_step_length > 0
        self._env = env
        self._step_length = robot_step_length

    def iterate(self):
        for rbt in self._env.robots:
            goals = self._env.feasible_goal_coordinates(rbt)

