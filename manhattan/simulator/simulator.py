from typing import Optional, Tuple, List, Set
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import matplotlib  # type: ignore
from os.path import isdir, isfile, join
from os import mkdir, makedirs
import json
import attr
import logging, coloredlogs

logger = logging.getLogger(__name__)
field_styles = {
    "filename": {"color": "green"},
    "levelname": {"bold": True, "color": "black"},
    "name": {"color": "blue"},
}
coloredlogs.install(
    level="INFO",
    fmt="[%(filename)s:%(lineno)d] %(name)s %(levelname)s - %(message)s",
    field_styles=field_styles,
)

from manhattan.environment.environment import ManhattanWorld
from manhattan.agent.agent import Robot, Beacon
from manhattan.geometry.TwoDimension import SE2Pose, Point2
from manhattan.measurement.range_measurement import RangeMeasurement
from manhattan.measurement.odom_measurement import OdomMeasurement
from manhattan.measurement.loop_closure import LoopClosure
from manhattan.noise_models.range_noise_model import (
    RangeNoiseModel,
    ConstantGaussianRangeNoiseModel as ConstGaussRangeSensor,
    VaryingMeanGaussianRangeNoiseModel as VaryGaussRangeSensor,
)
from manhattan.noise_models.odom_noise_model import (
    OdomNoiseModel,
    GaussianOdomNoiseModel as GaussOdomSensor,
)
from manhattan.noise_models.loop_closure_model import (
    LoopClosureModel,
    GaussianLoopClosureModel as GaussLoopClosureSensor,
)
from manhattan.utils.sample_utils import choice
from manhattan.utils.attrib_utils import (
    probability_validator,
    positive_float_validator,
    positive_int_validator,
    positive_int_tuple_validator,
)
from py_factor_graph.utils.name_utils import get_robot_char_from_number
from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.variables import PoseVariable2D, LandmarkVariable2D
from py_factor_graph.measurements import (
    PoseMeasurement2D,
    FGRangeMeasurement,
    AmbiguousPoseMeasurement2D,
    AmbiguousFGRangeMeasurement,
)
from py_factor_graph.priors import PosePrior2D, LandmarkPrior2D


@attr.s(frozen=True, auto_attribs=True)
class SimulationParams:
    """
    Args:
        num_robots (int): Number of robots to simulate
        num_beacons (int): Number of beacons to simulate
        grid_shape (Tuple[int, int]): (rows, cols) the shape of the manhattan
            world
        y_steps_to_intersection (int): how many rows between each intersection
            where the robot can turn
        x_steps_to_intersection (int): how many columns between each
            intersection where the robot can turn
        cell_scale (float): the length of the sides of the cells in the
            manhattan world
        range_sensing_radius (float): the radius of the sensor
        range_sensing_prob (float): the probability of range sensing
        false_range_data_association_prob (float): the probability that the
            data association is incorrect
        outlier_prob (float): the probability that the measurement is an
            outlier
        max_num_loop_closures (int): the maximum number of loop closures to
            allow in a simulation
        loop_closure_prob (float): the probability that a loop closure is
            detected
        loop_closure_radius (float): the radius of the circle that is used
            to try to detect loop closures
        false_loop_closure_prob (float): the probability that the data
            association is incorrect for a given loop closure
        range_stddev (float): the standard deviation of the gaussian noise
            added to the range measurements
        odom_x_stddev (float): the standard deviation of the gaussian noise
            added to the x position of the odometry
        odom_y_stddev (float): the standard deviation of the gaussian noise
            added to the y position of the odometry
        odom_theta_stddev (float): the standard deviation of the gaussian
            noise added to the theta position of the odometry
        loop_x_stddev (float): the standard deviation of the gaussian noise
            added to the x position of the loop closures
        loop_y_stddev (float): the standard deviation of the gaussian noise
            added to the y position of the loop closures
        loop_theta_stddev (float): the standard deviation of the gaussian
            noise added to the theta position of the loop closures
        seed_num (int): the seed for the random number generator
        debug_mode (bool): whether to print debug information and run debugging
            checks
        groundtruth_measurements (bool): whether to use ground truth as the
            measured values regardless of noise model
        no_loop_pose_idx (List[int]): array of pose indices for which no loop
            closures will be generated
        exclude_last_n_poses_for_loop_closure (int): default is 2; exclude last
            n poses from LC candidates
    """

    num_robots: int = attr.ib(default=1, validator=positive_int_validator)
    num_beacons: int = attr.ib(default=0, validator=positive_int_validator)
    grid_shape: Tuple[int, int] = attr.ib(
        default=(10, 10), validator=positive_int_tuple_validator
    )
    y_steps_to_intersection: int = attr.ib(default=1, validator=positive_int_validator)
    x_steps_to_intersection: int = attr.ib(default=1, validator=positive_int_validator)
    cell_scale: float = attr.ib(default=1.0, validator=positive_float_validator)
    range_sensing_prob: float = attr.ib(default=0.5, validator=probability_validator)
    range_sensing_radius: float = attr.ib(
        default=1.0, validator=positive_float_validator
    )
    false_range_data_association_prob: float = attr.ib(
        default=0.2, validator=probability_validator
    )
    outlier_prob: float = attr.ib(default=0.1, validator=probability_validator)
    max_num_loop_closures: int = attr.ib(default=2, validator=positive_int_validator)
    loop_closure_prob: float = attr.ib(default=0.5, validator=probability_validator)
    loop_closure_radius: float = attr.ib(default=10, validator=positive_float_validator)
    false_loop_closure_prob: float = attr.ib(
        default=0.2, validator=probability_validator
    )
    range_stddev: float = attr.ib(default=5, validator=positive_float_validator)
    odom_x_stddev: float = attr.ib(default=1e-1, validator=positive_float_validator)
    odom_y_stddev: float = attr.ib(default=1e-1, validator=positive_float_validator)
    odom_theta_stddev: float = attr.ib(default=1e-2, validator=positive_float_validator)
    loop_x_stddev: float = attr.ib(default=1e-1, validator=positive_float_validator)
    loop_y_stddev: float = attr.ib(default=1e-1, validator=positive_float_validator)
    loop_theta_stddev: float = attr.ib(default=1e-2, validator=positive_float_validator)
    seed_num: int = attr.ib(default=0, validator=positive_int_validator)
    debug_mode: bool = attr.ib(default=False)
    groundtruth_measurements: bool = attr.ib(default=False)
    no_loop_pose_idx: List[int] = attr.ib(default=[])
    exclude_last_n_poses_for_loop_closure: int = attr.ib(
        default=2, validator=positive_int_validator
    )


class ManhattanSimulator:
    """This class defines a simulator using Manhattan world-like environments.
    The simulator class keeps track of the state of the robots and beacons and
    provides a interface to everything needed to perform actions and get measurements.
    """

    @staticmethod
    def check_simulation_params(sim_params: SimulationParams) -> None:
        """Checks the validity of the provided simulation parameters.

        Args:
            sim_params (SimulationParams): the simulation parameters to check

        """
        if sim_params.groundtruth_measurements:
            print("WARNING: groundtruth_measurements is set to True. ")

        # check input arguments for simulation parameters

        assert sim_params.num_robots > 0, "num_robots must be greater than 0"
        assert sim_params.num_beacons >= 0, "num_beacons cannot be negative"

        # grid_shape is tuple of positive integers
        assert len(sim_params.grid_shape) == 2
        assert all(0 < x for x in sim_params.grid_shape)

        if sim_params.num_beacons > 0:
            assert (
                sim_params.y_steps_to_intersection > 1
                and sim_params.x_steps_to_intersection > 1
            ), "Need some space in grid to place beacons"

        # row and column spacing evenly fits into the grid shape
        assert sim_params.grid_shape[0] % sim_params.y_steps_to_intersection == 0
        assert sim_params.grid_shape[1] % sim_params.x_steps_to_intersection == 0

        # row_intersection_number is int > 0 and <= grid_shape[0]
        assert 0 <= sim_params.y_steps_to_intersection <= sim_params.grid_shape[0]

        # column_intersection_number is int > 0 and <= grid_shape[1]
        assert 0 <= sim_params.x_steps_to_intersection <= sim_params.grid_shape[1]

        # cell_scale is positive float
        assert sim_params.cell_scale > 0

        # range_sensing_prob is float between 0 and 1
        assert 0 <= sim_params.range_sensing_prob <= 1

        # range_sensing_radius is float > 0
        assert 0 < sim_params.range_sensing_radius

        # false_range_data_association_prob is float between 0 and 1
        assert 0 <= sim_params.false_range_data_association_prob <= 1

        # outlier_prob is float between 0 and 1
        assert 0 <= sim_params.outlier_prob <= 1

        # loop_closure_prob is float between 0 and 1
        assert 0 <= sim_params.loop_closure_prob <= 1

        # loop_closure_radius is float > cell_scale
        assert sim_params.cell_scale < sim_params.loop_closure_radius

        # false_loop_closure_prob is float between 0 and 1
        assert 0 <= sim_params.false_loop_closure_prob <= 1

        # all stddevs are positive floats
        assert 0 < sim_params.range_stddev
        assert 0 < sim_params.odom_x_stddev
        assert 0 < sim_params.odom_y_stddev
        assert 0 < sim_params.odom_theta_stddev
        assert 0 < sim_params.loop_x_stddev
        assert 0 < sim_params.loop_y_stddev
        assert 0 < sim_params.loop_theta_stddev

    def check_simulation_state(
        self,
    ) -> None:
        """Does some simple checking to make sure everything is in order in the
        simulation
        """

        if self.sim_params.debug_mode == False:
            return

        # check that robot and beacon lists are constructed correctly
        for true_pose_chain in self._groundtruth_poses:
            assert len(true_pose_chain) == (self.timestep) + 1

    def __init__(self, sim_params: SimulationParams) -> None:

        # run a bunch of checks to make sure input is valid
        self.check_simulation_params(sim_params)
        np.random.seed(sim_params.seed_num)

        if sim_params.groundtruth_measurements:
            logger.warning("Groundtruth measurements are enabled.")

        self._env = ManhattanWorld(
            grid_vertices_shape=sim_params.grid_shape,
            y_steps_to_intersection=sim_params.y_steps_to_intersection,
            x_steps_to_intersection=sim_params.x_steps_to_intersection,
            cell_scale=sim_params.cell_scale,
        )
        self._sim_params = sim_params
        self._robots: List[Robot] = []
        self._beacons: List[Beacon] = []

        self._timestep = 0

        # bookkeeping for pose measurements
        self._num_loop_closures = 0
        self._groundtruth_poses: List[List[SE2Pose]] = []

        # bookkeeping for range measurements
        self._sensed_beacons: Set[Beacon] = set()

        ### measurement models
        # range measurements
        self._base_range_model = ConstGaussRangeSensor(
            mean=0.0, stddev=self._sim_params.range_stddev
        )

        # odometry measurements
        odom_cov_x = self._sim_params.odom_x_stddev ** 2
        odom_cov_y = self._sim_params.odom_y_stddev ** 2
        odom_cov_theta = self._sim_params.odom_theta_stddev ** 2
        self._base_odometry_model = GaussOdomSensor(
            mean=np.zeros(3),
            covariance=np.diag([odom_cov_x, odom_cov_y, odom_cov_theta]),
        )

        # loop closures
        loop_cov_x = self._sim_params.loop_x_stddev ** 2
        loop_cov_y = self._sim_params.loop_y_stddev ** 2
        loop_cov_theta = self._sim_params.loop_theta_stddev ** 2
        self._base_loop_closure_model = GaussLoopClosureSensor(
            mean=np.zeros(3),
            covariance=np.diag([loop_cov_x, loop_cov_y, loop_cov_theta]),
        )

        # Add factor graph structure to hold data
        self._factor_graph = FactorGraphData(dimension=2)

        # * add these after everything else is initialized
        self.add_robots(sim_params.num_robots)
        self.add_beacons(sim_params.num_beacons)

        # make sure everything constructed correctly
        self.check_simulation_state()

        # for visualizing things
        self._robot_plot_objects = []  # type: ignore
        self._beacon_plot_objects = []  # type: ignore
        self.fig, self.ax = plt.subplots()
        x_lb, y_lb, x_ub, y_ub = self._env.bounds
        self.ax.set_xlim(x_lb - 1, x_ub + 1)
        self.ax.set_ylim(y_lb - 1, y_ub + 1)

    # make a destructor to close the plot
    def __del__(self) -> None:
        plt.close(self.fig)

    def __str__(self):
        line = "Simulator Environment\n"
        line += f"Sim Params: {self.sim_params}\n"
        line += f"Timestep: {self._timestep}\n"

    @property
    def robots(self) -> List[Robot]:
        return self._robots

    @property
    def beacons(self) -> List[Beacon]:
        return self._beacons

    @property
    def timestep(self) -> int:
        return self._timestep

    @property
    def num_robots(self) -> int:
        return len(self._robots)

    @property
    def num_beacons(self) -> int:
        return len(self._beacons)

    @property
    def sim_params(self) -> SimulationParams:
        return self._sim_params

    ###### Simulation interface methods ######

    def save_simulation_data(
        self, data_dir: str, format: str = "fg", filename: str = "factor_graph"
    ) -> str:
        """Saves the simulation data to a file with a given format.

        Args:
            data_dir (str): where to save the data to
            format (str, optional): the format of the data. Defaults to "efg".

        Returns:
            (str): the filepath
        """
        if not isdir(data_dir):
            makedirs(data_dir)

        # save a .json with all of the simulation parameters
        with open(data_dir + "/params.json", "w") as f:
            json.dump(attr.asdict(self.sim_params), f, indent=4)

        # save the simulation data to file
        filepath = join(data_dir, f"{filename}.{format}")
        self._factor_graph.save_to_file(filepath)

        return filepath

    def animate_odometry(self, show_gt: bool = False) -> None:
        """Visualizes the odometry data for the simulation.

        Args:
            show_gt (bool, optional): whether to show the ground truth. Defaults to False.
        """
        self._factor_graph.animate_odometry(show_gt=True, pause=0.01)

    def random_step(self) -> None:
        self._move_robots_randomly()
        self._update_range_measurements()
        self._update_loop_closures()

        # make sure everything was filled in correctly
        self.check_simulation_state()

    def execute_trajectories(self, trajectories: List[List[Tuple[int, int]]]):
        raise NotImplementedError

    def add_robots(self, num_robots: int) -> None:
        for _ in range(num_robots):
            self.add_robot()

    def add_beacons(self, num_beacons: int) -> None:
        for _ in range(num_beacons):
            self.add_beacon()

    def add_robot(
        self,
        start_pose: Optional[SE2Pose] = None,
        range_model: Optional[RangeNoiseModel] = None,
        odom_model: Optional[OdomNoiseModel] = None,
        loop_closure_model: Optional[LoopClosureModel] = None,
    ) -> None:
        """Add a robot to the simulator. If no pose is provided, a random pose
        is sampled from the environment.

        Args:
            start_pose (SE2Pose, optional): where to add the robot. Defaults to None.
            range_model (RangeNoiseModel, optional): the robot's range sensing
                model. Defaults to None.
            odom_model (OdomNoiseModel, optional): the robot's odometry model.
                Defaults to None.
            LoopClosureModel (LoopClosureModel, optional): the robot's loop
                closure model. Defaults to None.
        """

        # if no pose passed in, sample a random pose
        num_existing_robots = len(self._robots)
        cur_robot_idx = num_existing_robots
        robot_name = get_robot_char_from_number(cur_robot_idx)
        if start_pose is None:
            frame_name = f"{robot_name}0"
            if num_existing_robots == 0:
                start_pose = SE2Pose(
                    0.0,
                    0.0,
                    0.0,
                    local_frame=frame_name,
                    base_frame="world",
                )
            else:
                start_pose = self._env.get_random_robot_pose(local_frame=frame_name)

        if range_model is None:
            range_model = self._base_range_model

        if odom_model is None:
            odom_model = self._base_odometry_model

        if loop_closure_model is None:
            loop_closure_model = self._base_loop_closure_model

        # make sure that robot pose abides by the rules of the environment
        assert self._env.pose_is_robot_feasible(
            start_pose
        ), f"Robot pose {start_pose} is not feasible"

        robot = Robot(
            robot_name, start_pose, range_model, odom_model, loop_closure_model
        )
        self._robots.append(robot)

        # add to lists to track measurements
        self._groundtruth_poses.append([])
        self._groundtruth_poses[-1].append(start_pose)
        assert all(len(x) == 1 for x in self._groundtruth_poses), (
            "Should only have starting poses when adding "
            "robots to make sure robots aren't added after simulator iterations"
        )

        # update factor graph with new pose
        pose_loc = (start_pose.x, start_pose.y)
        pose_theta = start_pose.theta
        self._factor_graph.add_pose_variable(
            PoseVariable2D(start_pose.local_frame, pose_loc, pose_theta)
        )

        # if first robot, add prior to pin
        if num_existing_robots == 0:
            translation_precision = 100.0
            rotation_precision = 1000.0
            pose_prior = PosePrior2D(
                start_pose.local_frame,
                pose_loc,
                pose_theta,
                translation_precision,
                rotation_precision,
            )
            self._factor_graph.add_pose_prior(pose_prior)

    def add_beacon(
        self,
        position: Optional[Point2] = None,
        range_model: RangeNoiseModel = ConstGaussRangeSensor(),
    ) -> None:
        """Add a beacon to the simulator. If no position is provided, a random
        position is sampled from the environment.

        Args:
            position (Point2, optional): the beacon's position. Defaults to None.
            range_model (RangeNoiseModel, optional): The beacon's range model.
                Defaults to ConstGaussRangeSensor().
        """

        if position is None:
            position = self._env.get_random_beacon_point(frame="world")
            if position is None:
                return

        # make sure that beacon position abides by the rules of the environment
        assert self._env.position_is_beacon_feasible(
            position
        ), f"Beacon position {position} is not feasible"

        name = f"L{len(self._beacons)}"
        beacon = Beacon(name, position, range_model)
        self._beacons.append(beacon)
        self._factor_graph.add_landmark_variable(
            LandmarkVariable2D(name, (position.x, position.y))
        )

    def increment_timestep(self) -> None:
        self._timestep += 1

    ###### Internal methods to move robots ######

    def _move_robots_randomly(
        self,
    ) -> None:
        """Randomly moves all the robots to a neighboring vertex and records the
        resulting odometry measurement

        Note: the robots are not allowed to turn around
        """
        self.increment_timestep()

        # iterate over all robots
        for robot_idx, robot in enumerate(self._robots):

            # get all possible vertices to move to (all adjacent vertices not
            # behind robot)
            possible_moves = self._env.get_neighboring_robot_vertices_not_behind_robot(
                robot
            )

            # remove any moves that would result in collision of robots
            for other_robot_idx, other_robot in enumerate(self._robots):
                if other_robot_idx == robot_idx:
                    continue

                for move_idx, move in enumerate(possible_moves):
                    if other_robot.position == move[0]:
                        possible_moves.pop(move_idx)
                        break

            # if no possible moves, we'll let the robot turn around
            if len(possible_moves) == 0:
                possible_moves.append(self._env.get_vertex_behind_robot(robot))

            # randomly select a move from the list
            move = choice(possible_moves)
            move_pt: Point2 = move[0]
            bearing: float = move[1]

            # get the move in the robot local frame
            move_pt_local = robot.pose.transform_base_point_to_local(move_pt)

            # frame name represents robot and timestep
            move_frame_name = f"{robot.name}{robot.timestep+1}"

            # represent the move as a pose
            move_transform = SE2Pose(
                move_pt_local.x,
                move_pt_local.y,
                bearing,
                local_frame=move_frame_name,
                base_frame=robot.pose.local_frame,
            )

            # move the robot and store the measurement and new pose
            odom_measurement = robot.move(
                move_transform, self.sim_params.groundtruth_measurements
            )

            cur_pose = robot.pose
            self._factor_graph.add_pose_variable(
                PoseVariable2D(
                    cur_pose.local_frame, (cur_pose.x, cur_pose.y), cur_pose.theta
                )
            )
            self._store_odometry_measurement(robot_idx, odom_measurement)

            # make sure nothing weird happened with the timesteps
            assert self.timestep == robot.timestep

    ###### Internal methods to add measurements to the simulator ######

    def _store_odometry_measurement(
        self, robot_idx: int, measurement: OdomMeasurement
    ) -> None:
        """Store a measurement from the robot's odometry.

        Args:
            robot_idx (int): index of the robot that made the measurement.
            measurement (OdomMeasurement): the measurement to store.
        """
        robot = self._robots[robot_idx]

        pose_measure = PoseMeasurement2D(
            measurement.base_frame,
            measurement.local_frame,
            measurement.delta_x,
            measurement.delta_y,
            measurement.delta_theta,
            measurement.translation_precision,
            measurement.rotation_precision,
        )
        self._factor_graph.add_odom_measurement(robot_idx, pose_measure)

        # store the groundtruth pose as well (for loop closures)
        self._groundtruth_poses[robot_idx].append(robot.pose)

    def _update_range_measurements(self) -> None:
        """Update the range measurements for each robot."""
        for cur_robot_id in range(self.num_robots):
            cur_robot = self._robots[cur_robot_id]

            # get all ranging to other robots
            for other_robot_id in range(cur_robot_id + 1, self.num_robots):
                assert cur_robot_id < other_robot_id

                other_robot = self._robots[other_robot_id]

                # get distance between robot and other_robot
                dist = cur_robot.distance_to_other_agent(other_robot)

                # TODO fix bug so that robots do not end up in same location and
                # do not have zero distances
                if dist < 1e-2:
                    continue

                if dist < self.sim_params.range_sensing_radius:
                    if np.random.random() < self.sim_params.range_sensing_prob:
                        measure = cur_robot.range_measurement_from_dist(
                            dist, self.sim_params.groundtruth_measurements
                        )
                        self._add_robot_to_robot_range_measurement(
                            cur_robot_id, other_robot_id, measure
                        )

            # get all ranging to beacons
            for beacon_id in range(self.num_beacons):

                beacon = self._beacons[beacon_id]

                # get distance between robot and other_robot
                dist = cur_robot.distance_to_other_agent(beacon)

                if dist < self.sim_params.range_sensing_radius:
                    if np.random.random() < self.sim_params.range_sensing_prob:
                        measure = cur_robot.range_measurement_from_dist(dist)
                        self._add_robot_to_beacon_range_measurement(
                            cur_robot_id, beacon_id, measure
                        )

    def _update_loop_closures(self) -> None:
        """Possibly add loop closures for each robot.

        Loop closures are of form (pose_1, pose_2)
        """

        # can definitely make this faster using numpy or something to
        # compute the distances between all pairs of poses
        if self._num_loop_closures >= self.sim_params.max_num_loop_closures:
            return

        for cur_robot_id in range(self.num_robots):

            # roll dice to see if we can get a loop closure here. If greater
            # than this value then no loop closure
            if np.random.rand() > self.sim_params.loop_closure_prob or (
                len(self._groundtruth_poses[cur_robot_id]) - 1
                in self._sim_params.no_loop_pose_idx
            ):
                continue

            cur_robot = self._robots[cur_robot_id]
            cur_pose = cur_robot.pose
            cur_x = cur_pose.x
            cur_y = cur_pose.y
            possible_loop_closures = []

            # gather up list of closure candidates
            for loop_clos_robot_id in range(self.num_robots):

                # ignore the two most recent poses, as it shouldn't be
                # considered for loop closures
                candidate_pose_chain = self._groundtruth_poses[loop_clos_robot_id][
                    : -self.sim_params.exclude_last_n_poses_for_loop_closure
                ]
                for cand_pose in candidate_pose_chain:

                    # get difference between the current pose and the candidate pose
                    cand_x = cand_pose.x
                    cand_y = cand_pose.y
                    diff_x = abs(cur_x - cand_x)
                    diff_y = abs(cur_y - cand_y)
                    x_too_far = diff_x > self.sim_params.loop_closure_radius
                    y_too_far = diff_y > self.sim_params.loop_closure_radius

                    # approximate the radius check just by a square
                    if x_too_far or y_too_far:
                        continue
                    else:
                        possible_loop_closures.append(cand_pose)

            if len(possible_loop_closures) > 0:
                true_loop_closure_pose = choice(possible_loop_closures)

                # if there are enough options and RNG says to make a fake loop
                # closure, we intentionally mess up the data association
                if (
                    len(possible_loop_closures) > 1
                    and np.random.rand() < self.sim_params.false_loop_closure_prob
                ):

                    # remove the true loop closure from the options and pick a
                    # new one for the measured data association
                    possible_loop_closures.remove(true_loop_closure_pose)
                    false_loop_closure_pose = choice(possible_loop_closures)

                    # record the false pose as the measured data association
                    loop_closure = cur_robot.get_loop_closure_measurement(
                        true_loop_closure_pose,
                        false_loop_closure_pose.local_frame,
                        self.sim_params.groundtruth_measurements,
                    )

                    # fill in the incorrect loop closure in factor graph
                    ambiguous_loop_closure = AmbiguousPoseMeasurement2D(
                        base_pose=loop_closure.base_frame,
                        measured_to_pose=false_loop_closure_pose.local_frame,
                        true_to_pose=true_loop_closure_pose.local_frame,
                        x=loop_closure.delta_x,
                        y=loop_closure.delta_y,
                        theta=loop_closure.delta_theta,
                        translation_precision=loop_closure.translation_precision,
                        rotation_precision=loop_closure.rotation_precision,
                    )
                    self._factor_graph.add_ambiguous_pose_measurement(
                        ambiguous_loop_closure
                    )

                else:
                    # otherwise just use the true loop closure
                    loop_closure = cur_robot.get_loop_closure_measurement(
                        true_loop_closure_pose,
                        true_loop_closure_pose.local_frame,
                        self.sim_params.groundtruth_measurements,
                    )

                    # fill in loop closure in factor graph
                    measure = PoseMeasurement2D(
                        loop_closure.base_frame,
                        loop_closure.local_frame,
                        loop_closure.delta_x,
                        loop_closure.delta_y,
                        loop_closure.delta_theta,
                        loop_closure.translation_precision,
                        loop_closure.rotation_precision,
                    )
                    self._factor_graph.add_loop_closure(measure)

                self._num_loop_closures += 1

    def _get_incorrect_robot_to_robot_range_association(
        self, robot_1_idx: int, robot_2_idx: int
    ) -> Tuple[str, str]:
        """returns an incorrect data association for the range measurement
        (either to an incorrect robot or to a beacon)

        Args:
            robot_1_idx (int): the index of the first robot
            robot_2_idx (int): the index of the other robot

        Returns:
            Tuple[str, str]: (robot_1_name, incorrect_name) The incorrect data
                association
        """
        assert 0 <= robot_1_idx < self.num_robots
        assert 0 <= robot_2_idx < self.num_robots
        assert robot_1_idx != robot_2_idx

        # first robot will always be correct?
        assoc_1 = self._robots[robot_1_idx].pose.local_frame

        # get all other robots
        true_other_assoc = self._robots[robot_2_idx].pose.local_frame
        robot_options = [x.pose.local_frame for x in self._robots]
        robot_options.remove(assoc_1)
        robot_options.remove(true_other_assoc)

        # get all possible beacon names
        beacon_options = [x.name for x in self._beacons]

        # concatenate all association options and randomly choose
        all_options = robot_options + beacon_options
        assoc_2 = choice(all_options)

        # robot_1_name and incorrect_data_association_name
        return (assoc_1, assoc_2)

    def _get_incorrect_robot_to_beacon_range_association(
        self, robot_idx: int, beacon_idx: int
    ) -> Tuple[str, str]:
        """returns an incorrect data association for the range measurement
        (either to an incorrect robot or to a beacon)

        Args:
            robot_idx (int): the true robot index
            beacon_idx (int): the true beacon index

        Returns:
            Tuple[str, str]: the incorrect data association
        """
        assert 0 <= robot_idx < self.num_robots
        assert 0 <= beacon_idx < self.num_beacons

        # robot will always be correct?
        assoc_1 = self._robots[robot_idx].pose.local_frame

        # get all other robots
        robot_options = [x.pose.local_frame for x in self._robots]
        robot_options.remove(assoc_1)

        # get all other beacons that have already been sensed
        true_beacon_name = self._beacons[beacon_idx].name
        beacon_options = [x.name for x in self._beacons if x in self._sensed_beacons]

        if true_beacon_name in beacon_options:
            beacon_options.remove(true_beacon_name)

        # if no other beacons have been sensed just return true association
        if len(beacon_options) == 0:
            return (assoc_1, true_beacon_name)

        # concatenate all association options and randomly choose
        all_options = robot_options + beacon_options
        assoc_2 = choice(all_options)

        # robot_1_name and incorrect_data_association_name
        return (assoc_1, assoc_2)

    def _add_robot_to_robot_range_measurement(
        self, robot_1_idx: int, robot_2_idx: int, measurement: RangeMeasurement
    ):
        """Add a new range measurement between two robots. Randomly chooses if
        the data association is incorrect. If incorrect, the association can be
        to robots or beacons

        Args:
            robot_1_idx (int): [description]
            robot_2_idx (int): [description]
            measurement (RangeMeasurement): [description]

        """
        assert 0 <= robot_1_idx < self.num_robots
        assert 0 <= robot_2_idx < self.num_robots
        assert robot_1_idx < robot_2_idx

        assert 0.0 <= measurement.true_distance <= self.sim_params.range_sensing_radius

        # fill in the measurement info
        true_association = (
            self._robots[robot_1_idx].pose.local_frame,
            self._robots[robot_2_idx].pose.local_frame,
        )

        # randomly sample to decide if this is an incorrect association
        if np.random.rand() < self.sim_params.false_range_data_association_prob:
            measurement_association = (
                self._get_incorrect_robot_to_robot_range_association(
                    robot_1_idx, robot_2_idx
                )
            )
            ambiguous_fg_measure = AmbiguousFGRangeMeasurement(
                true_association,
                measurement_association,
                measurement.measured_distance,
                measurement.stddev,
            )
            self._factor_graph.add_ambiguous_range_measurement(ambiguous_fg_measure)
        else:
            fg_measure = FGRangeMeasurement(
                true_association, measurement.measured_distance, measurement.stddev
            )
            self._factor_graph.add_range_measurement(fg_measure)

    def _add_robot_to_beacon_range_measurement(
        self, robot_idx: int, beacon_idx: int, measurement: RangeMeasurement
    ) -> None:
        """Add a new range measurement between a robot and a beacon. Randomly
        chooses if the data association is incorrect. If incorrect, the
        association can be to robots or beacons

        Args:
            robot_idx (int): the robot index
            beacon_idx (int): the beacon index
            measurement (RangeMeasurement): the measurement between the robot
                and the beacon
        """
        assert 0 <= robot_idx < self.num_robots
        assert 0 <= beacon_idx < self.num_beacons

        # fill in the measurement info
        true_association = (
            self._robots[robot_idx].pose.local_frame,
            self._beacons[beacon_idx].name,
        )
        self._sensed_beacons.add(self._beacons[beacon_idx])

        # randomly sample to see if this is a false data association range measurement
        if np.random.rand() < self.sim_params.false_range_data_association_prob:
            measurement_association = (
                self._get_incorrect_robot_to_beacon_range_association(
                    robot_idx, beacon_idx
                )
            )
            ambiguous_fg_measure = AmbiguousFGRangeMeasurement(
                true_association,
                measurement_association,
                measurement.measured_distance,
                measurement.stddev,
            )
            self._factor_graph.add_ambiguous_range_measurement(ambiguous_fg_measure)
        else:
            fg_measure = FGRangeMeasurement(
                true_association, measurement.measured_distance, measurement.stddev
            )
            self._factor_graph.add_range_measurement(fg_measure)

    #### print state of simulator methods ####

    def print_simulator_state(self):
        print(f"Timestep: {self._timestep}")
        self.print_robot_states()
        self.print_beacon_states()

    def print_robot_states(self):
        for robot in self.robots:
            print(robot)

    def print_beacon_states(self):
        for beacon in self._beacons:
            print(beacon)

    #### visualize simulator state methods ####

    def plot_grid(self):
        self._env.plot_environment(self.ax)

    def plot_beacons(self):
        """Plots all of the beacons"""
        assert len(self._beacon_plot_objects) == 0, (
            "Should not be plotting over existing beacons."
            + " This function should only be called once."
        )

        for i, beacon in enumerate(self._beacons):
            beacon_plot_obj = beacon.plot()
            self._beacon_plot_objects.append(beacon_plot_obj[0])

    def plot_robot_states(self):
        """Plots the current robot states"""

        # delete all of the already shown robot poses from the plot
        # this allows us to more efficiently update the animation
        for robot_plot_obj in self._robot_plot_objects:
            if robot_plot_obj in self.ax.lines:
                self.ax.lines.remove(robot_plot_obj)

        self._robot_plot_objects.clear()

        for i, robot in enumerate(self._robots):
            rob_plot_obj = robot.plot()
            self._robot_plot_objects.append(rob_plot_obj[0])

    def show_plot(self, animation: bool = False):
        """shows everything that's been plotted

        Args:
            animation (bool): if True, just gives a minor pause. If False,
                shows the plot and waits for the user to close it.
        """
        if animation:
            plt.pause(0.3)
        else:
            plt.show(block=True)
            self._robot_plot_objects.clear()
            self._beacon_plot_objects.clear()

            self.fig, self.ax = plt.subplots()
            x_lb, y_lb, x_ub, y_ub = self._env.bounds
            self.ax.set_xlim(x_lb - 1, x_ub + 1)
            self.ax.set_ylim(y_lb - 1, y_ub + 1)

    def close_plot(self):
        plt.close()
        self._robot_plot_objects.clear()
        self._beacon_plot_objects.clear()
