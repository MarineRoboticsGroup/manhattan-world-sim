from Environment import *
import random

class LocalPathPlanner:
    pass

class SimulationArgs:
    def __init__(self,
                 range_sensing_prob = .5,
                 ambiguous_data_association_prob = None,
                 outlier_prob = None,
                 loop_closure_prob = None,
                 loop_closure_radius = None):
        self.range_prob = range_sensing_prob
        self.lc_prob = loop_closure_prob
        self.lc_radius = loop_closure_radius

        # Apply to range measurements and loop closures
        self.ada_prob = ambiguous_data_association_prob
        self.outlier_prob = outlier_prob

class ManhattanSimulator:
    """This class defines a simulator using Manhattan world-like environments
    :param
    env: simulated environment in which agents are added.
    """
    def __init__(self, env: ManhattanWaterworld,
                 args: SimulationArgs
                 ):
        self._env = env
        self._args = args
        self._rbt2gtpose = {}
        for rbt in env.robots:
            self._rbt2gtpose[rbt] = [env._rbt2pose[rbt]]

    def execute_waypoints(self, waypoints):
        raise NotADirectoryError

    def random_waypoint_iterate(self):
        env = self._env
        for rbt in env.robots:
            cur_pose = env._rbt2pose[rbt]
            goals = env.nearest_robot_vertex_coordinates(cur_pose.x, cur_pose.y)
            next_wp = random.choice(goals)
            moves = rbt.local_path_planner(cur_pose=cur_pose, goal=Point2(*next_wp))
            pass


