from src.environment.environment import ManhattanWorld
from collections import namedtuple
import random


class ManhattanSimulator:
    """This class defines a simulator using Manhattan world-like environments.
    The simulator class keeps track of the state of the robots and beacons and
    provides a interface to everything needed to perform actions and get measurements.
    """

    simulation_params = namedtuple(
        "Manhattan Simulation Parameters",
        [
            "range_sensing_prob",
            "ambiguous_data_association_prob",
            "outlier_prob",
            "loop_closure_prob",
            "loop_closure_radius",
        ],
        defaults = (0.5, None, None, None, None)
    )

    def __init__(self, env: ManhattanWaterworld, args: simulation_params):
        assert isinstance(env, ManhattanWorld)
        assert isinstance(args, self.simulation_params)

        self._env = env
        self._args = args
        self._rbt2gtpose = {}
        for rbt in env.robots:
            self._rbt2gtpose[rbt] = [env._robot_poses[rbt]]

    def execute_waypoints(self, waypoints):
        raise NotADirectoryError

    def random_waypoint_iterate(self):
        env = self._env
        for rbt in env.robots:
            cur_pose = env._robot_poses[rbt]
            goals = env.nearest_robot_vertex_coordinates(cur_pose.x, cur_pose.y)
            next_wp = random.choice(goals)
            moves = rbt.local_path_planner(cur_pose=cur_pose, goal=Point2(*next_wp))
            pass
