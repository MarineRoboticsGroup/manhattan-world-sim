from typing import NamedTuple, Tuple, List
import random
import matplotlib.pyplot as plt

from manhattan.environment.environment import ManhattanWorld
from manhattan.agent.agent import Robot, Beacon
from manhattan.geometry.TwoDimension import SE2Pose, Point2
from manhattan.noise_models.range_noise_model import (
    RangeNoiseModel,
    ConstantGaussianRangeNoiseModel as ConstGaussRangeSensor,
    VaryingMeanGaussianRangeNoiseModel as VaryGaussRangeSensor,
)
from manhattan.noise_models.odom_noise_model import (
    OdomNoiseModel,
    GaussianOdomNoiseModel as GaussOdomSensor,
)


class ManhattanSimulator:
    """This class defines a simulator using Manhattan world-like environments.
    The simulator class keeps track of the state of the robots and beacons and
    provides a interface to everything needed to perform actions and get measurements.
    """

    class SimulationParams(NamedTuple):
        """
        Args:
            grid_shape (Tuple[int, int]): (rows, cols) the shape of the manhattan world
            row_corner_number (int): how many rows between each intersection
                where the robot can turn
            column_corner_number (int): how many columns between each
                intersection where the robot can turn
            cell_scale (float): the length of the sides of the cells in the
                manhattan world
            range_sensing_prob (float): the probability of range sensing
            ambiguous_data_association_prob (float): the probability that the
                data association is incorrect
            outlier_prob (float): the probability that the measurement is an
                outlier
            loop_closure_prob (float): the probability that a loop closure is
                detected
            loop_closure_radius (float): the radius of the circle that is used
                to try to detect loop closures
        """

        grid_shape: Tuple = (10, 10)
        row_corner_number: int = 1
        column_corner_number: int = 1
        cell_scale: float = 1.0
        range_sensing_prob: float = 0.5
        ambiguous_data_association_prob: float = 0.1
        outlier_prob: float = 0.1
        loop_closure_prob: float = 0.1
        loop_closure_radius: float = 2.0

    def __init__(self, sim_params: SimulationParams):

        # check input arguments for simulation parameters
        assert isinstance(sim_params, self.SimulationParams)

        # grid_shape is tuple of positive integers
        assert isinstance(sim_params.grid_shape, tuple)
        assert len(sim_params.grid_shape) == 2
        assert all(isinstance(x, int) for x in sim_params.grid_shape)
        assert all(0 < x for x in sim_params.grid_shape)

        # row and column spacing evenly fits into the grid shape
        assert sim_params.grid_shape[0] % sim_params.row_corner_number == 0
        assert sim_params.grid_shape[1] % sim_params.column_corner_number == 0

        # row_intersection_number is int > 0 and <= grid_shape[0]
        assert isinstance(sim_params.row_corner_number, int)
        assert 0 <= sim_params.row_corner_number <= sim_params.grid_shape[0]

        # column_intersection_number is int > 0 and <= grid_shape[1]
        assert isinstance(sim_params.column_corner_number, int)
        assert 0 <= sim_params.column_corner_number <= sim_params.grid_shape[1]

        # cell_scale is positive float
        assert isinstance(sim_params.cell_scale, float)
        assert sim_params.cell_scale > 0

        # range_sensing_prob is float between 0 and 1
        assert isinstance(sim_params.range_sensing_prob, float)
        assert 0 <= sim_params.range_sensing_prob <= 1

        # ambiguous_data_association_prob is float between 0 and 1
        assert isinstance(sim_params.ambiguous_data_association_prob, float)
        assert 0 <= sim_params.ambiguous_data_association_prob <= 1

        # outlier_prob is float between 0 and 1
        assert isinstance(sim_params.outlier_prob, float)
        assert 0 <= sim_params.outlier_prob <= 1

        # loop_closure_prob is float between 0 and 1
        assert isinstance(sim_params.loop_closure_prob, float)
        assert 0 <= sim_params.loop_closure_prob <= 1

        # loop_closure_radius is float > 0
        assert isinstance(sim_params.loop_closure_radius, float)
        assert 0 < sim_params.loop_closure_radius

        self._env = ManhattanWorld(
            grid_vertices_shape=sim_params.grid_shape,
            row_corner_number=sim_params.row_corner_number,
            column_corner_number=sim_params.column_corner_number,
            cell_scale=sim_params.cell_scale,
        )
        self._sim_params = sim_params
        self._robots = []
        self._beacons = []

    @property
    def robots(self) -> List[Robot]:
        return self._robots

    @property
    def beacons(self) -> List[Beacon]:
        return self._beacons

    def add_robots(self, num_robots: int):
        assert isinstance(num_robots, int)
        assert num_robots > 0
        for _ in range(num_robots):
            self.add_robot()

    def add_beacons(self, num_beacons: int):
        assert isinstance(num_beacons, int)
        assert num_beacons > 0
        for _ in range(num_beacons):
            self.add_beacon()

    def add_robot(
        self,
        start_pose: SE2Pose = None,
        range_model: RangeNoiseModel = ConstGaussRangeSensor(),
        odom_model: OdomNoiseModel = GaussOdomSensor(),
    ):
        """Add a robot to the simulator. If no pose is provided, a random pose
        is sampled from the environment.

        Args:
            start_pose (SE2Pose, optional): where to add the robot. Defaults to None.
            range_model (RangeNoiseModel, optional): the robot's range sensing
                model. Defaults to ConstGaussRangeSensor().
            odom_model (OdomNoiseModel, optional): the robot's odometry model.
                Defaults to GaussOdomSensor().
        """

        assert isinstance(start_pose, SE2Pose) or start_pose is None
        assert isinstance(range_model, RangeNoiseModel)
        assert isinstance(odom_model, OdomNoiseModel)

        # if no pose passed in, sample a random pose
        name = f"Robot {len(self._robots)}"
        if start_pose is None:
            frame_name = f"{name} time: 0"
            start_pose = self._env.get_random_robot_pose(local_frame=frame_name)

        # make sure that robot pose abides by the rules of the environment
        assert isinstance(start_pose, SE2Pose)
        assert self._env.pose_is_robot_feasible(
            start_pose
        ), f"Robot pose {start_pose} is not feasible"

        robot = Robot(name, start_pose, range_model, odom_model)
        self._robots.append(robot)

    def add_beacon(
        self,
        position: Point2 = None,
        range_model: RangeNoiseModel = ConstGaussRangeSensor(),
    ):
        """Add a beacon to the simulator. If no position is provided, a random
        position is sampled from the environment.

        Args:
            position (Point2, optional): the beacon's position. Defaults to None.
            range_model (RangeNoiseModel, optional): The beacon's range model.
                Defaults to ConstGaussRangeSensor().
        """
        assert isinstance(position, Point2) or position is None
        assert isinstance(range_model, RangeNoiseModel)

        if position is None:
            position = self._env.get_random_beacon_point(frame="world")
            if position is None:
                return

        # make sure that beacon position abides by the rules of the environment
        assert self._env.position_is_beacon_feasible(
            position
        ), f"Beacon position {position} is not feasible"

        name = f"Beacon {len(self._beacons)}"
        beacon = Beacon(name, position, range_model)
        self._beacons.append(beacon)

    def print_simulator_state(self):
        self.print_robot_states()
        self.print_beacon_states()

    def print_robot_states(self):
        for robot in self.robots:
            print(robot)

    def print_beacon_states(self):
        for beacon in self._beacons:
            print(beacon)

    def plot_current_state(self, show_grid=False):
        """Plots the current state of the simulator.

        Args:
            show_grid (bool, optional): whether to show the grid. Defaults to
            False.
        """

        if show_grid:
            self._env.plot_environment()

        for robot in self._robots:
            robot.plot()
        for beacon in self._beacons:
            beacon.plot()

        x_lb, y_lb, x_ub, y_ub = self._env.bounds
        plt.xlim(x_lb - 1, x_ub + 1)
        plt.ylim(y_lb - 1, y_ub + 1)

        plt.show(block=False)
        plt.pause(0.05)
        plt.clf()

    # TODO docstring this function
    def move_robots_randomly(self,):
        for robot in self._robots:
            possible_moves = self._env.get_neighboring_robot_vertices_not_behind_robot(
                robot
            )
            assert isinstance(possible_moves, list)
            assert all(
                isinstance(v[0], Point2) and isinstance(v[1], float)
                for v in possible_moves
            )
            assert len(possible_moves) > 0

            move = random.choice(possible_moves)
            move_pt, bearing = move

            move_pt_local = robot.pose.transform_base_point_to_local(move_pt)
            move_frame_name = f"{robot.name} time: {robot.timestep+1}"

            move_transform = SE2Pose(
                move_pt_local.x,
                move_pt_local.y,
                bearing,
                local_frame=move_frame_name,
                base_frame=robot.pose.local_frame,
            )
            robot.move(move_transform)


    def execute_trajectories(self, trajectories: List[List[Tuple[int, int]]]):
        raise NotImplementedError
