from collections import namedtuple
import random

from src.environment.environment import ManhattanWorld
from src.agent.agent import RobotAgent, BeaconAgent
from src.geometry.TwoDimension import SE2Pose, Point2
from src.noise_models.range_noise_model import RangeNoiseModel
from src.noise_models.odom_noise_model import OdomNoiseModel


class ManhattanSimulator:
    """This class defines a simulator using Manhattan world-like environments.
    The simulator class keeps track of the state of the robots and beacons and
    provides a interface to everything needed to perform actions and get measurements.
    """

    simulation_params = namedtuple(
        """
        Args:
            range_sensing_prob (float): the probability of range sensing
            row_intersection_number (int): how many rows between each intersection
                where the robot can turn
            column_intersection_number (int): how many columns between each
                intersection where the robot can turn
            ambiguous_data_association_prob (float): the probability that the
                data association is incorrect
            outlier_prob (float): the probability that the measurement is an
                outlier
            loop_closure_prob (float): the probability that a loop closure is
                detected
            loop_closure_radius (float): the radius of the circle that is used
                to try to detect loop closures
        """
        "Manhattan Simulation Parameters",
        [
            "range_sensing_prob",
            "row_corner_number",
            "column_corner_number",
            "ambiguous_data_association_prob",
            "outlier_prob",
            "loop_closure_prob",
            "loop_closure_radius",
        ],
        defaults=(0.5, None, None, None, None),
    )

    def __init__(self, env: ManhattanWaterworld, args: simulation_params):
        assert isinstance(env, ManhattanWorld)
        assert isinstance(args, self.simulation_params)

        self._env = env
        self._args = args
        self._robots = []
        self._beacons = []

    def _add_robot(
        self,
        start_pose: SE2Pose,
        range_model: RangeNoiseModel,
        odom_model: OdomNoiseModel,
    ):
        """Adds robot to simulator.

        Args:
            start_pose (SE2Pose): the starting pose of the robot
            range_model (RangeNoiseModel): the robot's range sensing model
            odom_model (OdomNoiseModel): the robot's odometry model
        """
        assert isinstance(start_pose, SE2Pose)
        assert isinstance(range_model, RangeNoiseModel)
        assert isinstance(odom_model, OdomNoiseModel)

        name = str(len(self._robots))
        robot = RobotAgent(name, start_pose, range_model, odom_model)
        assert isinstance(robot, RobotAgent)
        self._robots.append(robot)

    def _add_beacon(
        self,
        position: Point2,
        range_model: RangeNoiseModel,
    ):
        """Adds a beacon to the simulator.

        Args:
            position (Point2): the beacon's position
            range_model (RangeNoiseModel): the beacon's range sensing model
        """
        assert isinstance(position, Point2)
        assert isinstance(range_model, RangeNoiseModel)

        name = str(len(self._beacons))
        beacon = BeaconAgent(name, position, range_model)
        assert isinstance(beacon, BeaconAgent)
        self._beacons.append(beacon)

    def move_robots_randomly(
        self,
    ):
        for robot in self._robots:
            # get robot position
            robot_loc = robot.position

            # get robot vertex
            robot_vert = self._env.nearest_robot_vertex_coordinates(robot_loc)
            i, j = robot_vert

            # check if robot is at an intersection
            is_at_row_intersection = i % self._args.row_corner_number == 0
            is_at_col_intersection = j % self._args.column_corner_number == 0
            assert isinstance(is_at_row_intersection, bool)  # just to be sure
            assert isinstance(is_at_col_intersection, bool)  # just to be sure

            # if robot is at an intersection then it can consider turning left
            # or right
            if is_at_row_intersection and is_at_col_intersection:
                neighbor_vertices = self._env.get_neighboring_robot_vertices(robot_vert)
                neighbor_vertices_coords = self._env.vertices2coordinates(
                    neighbor_vertices
                )

                move_dist = self._env.scale
                # left_turn = SE2Pose(
                # move_options =

                raise NotImplementedError
            # get neighbors of vertex
            # select n
            # rbt.random_move()
        raise NotImplementedError

    def execute_trajectories(self, trajectories: List[List[Tuple[int, int]]]):
        raise NotImplementedError

    def random_waypoint_iterate(self):
        env = self._env
        for rbt in env.robots:
            cur_pose = env._robot_poses[rbt]
            goals = env.nearest_robot_vertex_coordinates(cur_pose.x, cur_pose.y)
            next_wp = random.choice(goals)
            moves = rbt.local_path_planner(cur_pose=cur_pose, goal=Point2(*next_wp))
            pass
