from typing import Optional, NamedTuple, Tuple, List, Set
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import matplotlib  # type: ignore
from os.path import isdir, isfile, join
from os import mkdir, makedirs
import json

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
from manhattan.simulator.save_file_utils import save_to_efg_format
from manhattan.utils.sample_utils import choice


class SimulationParams(NamedTuple):
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
        no_loop_pose_idx (list): array of pose indices for which no loop closures will be generated
        exclude_last_n_poses_for_loop_closure (int): default is 2; exclude last n poses from LC candidates
    """

    num_robots: int = 1
    num_beacons: int = 0
    grid_shape: Tuple = (10, 10)
    y_steps_to_intersection: int = 1
    x_steps_to_intersection: int = 1
    cell_scale: float = 1.0
    range_sensing_prob: float = 0.5
    range_sensing_radius: float = 5.0
    false_range_data_association_prob: float = 0.3
    outlier_prob: float = 0.1
    max_num_loop_closures: int = 100
    loop_closure_prob: float = 0.1
    loop_closure_radius: float = 2.0
    false_loop_closure_prob: float = 0.1
    range_stddev: float = 0.1
    odom_x_stddev: float = 0.1
    odom_y_stddev: float = 0.1
    odom_theta_stddev: float = 0.1
    loop_x_stddev: float = 0.1
    loop_y_stddev: float = 0.1
    loop_theta_stddev: float = 0.1
    seed_num: int = 0
    debug_mode: bool = False
    groundtruth_measurements: bool = False
    no_loop_pose_idx: List = []
    exclude_last_n_poses_for_loop_closure: int = 2


# TODO integrate the probabilities of measurements into the simulator
# data association probability
# outlier probability
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
        # check input arguments for simulation parameters

        assert sim_params.num_robots > 0, "num_robots must be greater than 0"
        assert sim_params.num_beacons >= 0, "num_beacons cannot be negative"

        # grid_shape is tuple of positive integers
        assert len(sim_params.grid_shape) == 2
        assert all(0 < x for x in sim_params.grid_shape)

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

        # single lists
        assert len(self._odom_measurements) == len(self._robots)

    def __init__(self, sim_params: SimulationParams) -> None:

        # run a bunch of checks to make sure input is valid
        self.check_simulation_params(sim_params)
        np.random.seed(sim_params.seed_num)

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

        # pose measurements
        self._odom_measurements: List[List[OdomMeasurement]] = []
        self._loop_closures: List[LoopClosure] = []
        self._groundtruth_poses: List[List[SE2Pose]] = []

        # range measurements
        self._range_measurements: List[RangeMeasurement] = []
        self._range_associations: List[Tuple[str, str]] = []
        self._groundtruth_range_associations: List[Tuple[str, str]] = []
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

        # add these after everything else is initialized
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

    def __str__(self):
        line = "Simulator Environment\n"
        line += f"Sim Params: {self.sim_params}\n"
        line += f"Timestep: {self._timestep}\n"

    @property
    def file_name(self) -> str:
        line = f"simEnvironment_"
        line += f"grid{self.sim_params.grid_shape[0]}x"
        line += f"{self.sim_params.grid_shape[1]}_"
        line += f"rowCorner{self.sim_params.y_steps_to_intersection}_"
        line += f"colCorner{self.sim_params.x_steps_to_intersection}_"
        line += f"cellScale{self.sim_params.cell_scale}_"
        line += f"rangeProb{self.sim_params.range_sensing_prob}_"
        line += f"rangeRadius{self.sim_params.range_sensing_radius:.0f}_"
        line += f"falseRangeProb{self.sim_params.false_range_data_association_prob}_"
        line += f"outlierProb{self.sim_params.outlier_prob}_"
        line += f"loopClosureProb{self.sim_params.loop_closure_prob}_"
        line += f"loopClosureRadius{self.sim_params.loop_closure_radius:.0f}_"
        line += f"falseLoopClosureProb{self.sim_params.false_loop_closure_prob}_"
        line += f"timestep{self._timestep}"
        line = line.replace(".", "")
        return line

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

    def save_simulation_data(self, data_dir: str, format: str = "efg") -> None:
        """Saves the simulation data to a file with a given format.

        Args:
            data_dir (str): where to save the data to
            format (str, optional): the format of the data. Defaults to "efg".
        """
        if not isdir(data_dir):
            makedirs(data_dir)

        if format == "efg":
            self._save_data_as_efg_format(data_dir)
            with open(data_dir + "/params.json", "w") as f:
                json.dump(self.sim_params._asdict(), f)
        else:
            raise NotImplementedError(f"Data format {format} is not supported.")

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
        name = f"Robot {num_existing_robots}"
        if start_pose is None:
            frame_name = f"{name} time: 0"
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
        assert isinstance(start_pose, SE2Pose)
        assert self._env.pose_is_robot_feasible(
            start_pose
        ), f"Robot pose {start_pose} is not feasible"

        robot = Robot(name, start_pose, range_model, odom_model, loop_closure_model)
        self._robots.append(robot)

        # add to lists to track measurements
        self._odom_measurements.append([])
        self._groundtruth_poses.append([])
        self._groundtruth_poses[-1].append(start_pose)

        # make sure lists are all correct sizes
        assert len(self._odom_measurements) == len(self._robots)

    def add_beacon(
        self,
        position: Point2 = None,
        range_model: RangeNoiseModel = ConstGaussRangeSensor(),
    ) -> None:
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

    def increment_timestep(self) -> None:
        self._timestep += 1

    ##### Internal methods to save data #####

    def _save_data_as_efg_format(self, data_dir: str) -> None:
        """Saves the data in the Extended Factor Graph (EFG) format.

        Args:
            data_dir (str): the directory to save everything in
        """
        save_file = f"{data_dir}/{self.file_name}.fg"
        save_to_efg_format(
            save_file,
            odom_measurements=self._odom_measurements,
            loop_closures=self._loop_closures,
            gt_poses=self._groundtruth_poses,
            beacons=self._beacons,
            range_measurements=self._range_measurements,
            range_associations=self._range_associations,
            gt_range_associations=self._groundtruth_range_associations,
        )
        print(f"Saved file to: {save_file}")

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
            move_frame_name = f"{robot.name} time: {robot.timestep+1}"

            # represent the move as a pose
            move_transform = SE2Pose(
                move_pt_local.x,
                move_pt_local.y,
                bearing,
                local_frame=move_frame_name,
                base_frame=robot.pose.local_frame,
            )

            # move the robot and store the measurement
            odom_measurement = robot.move(
                move_transform, self.sim_params.groundtruth_measurements
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

        # add measurement and timestamp
        self._odom_measurements[robot_idx].append(measurement)

        # store the groundtruth pose as well
        self._groundtruth_poses[robot_idx].append(robot.pose)

        # should always have one more pose than odometry measurement
        assert len(self._groundtruth_poses[robot_idx]) - 1 == len(
            self._odom_measurements[robot_idx]
        ), f"{len(self._groundtruth_poses[robot_idx]) - 1} groundtruth poses vs {len(self._odom_measurements[robot_idx])}"

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

                if dist < self.sim_params.range_sensing_radius:
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

        if len(self._loop_closures) >= self.sim_params.max_num_loop_closures:
            return

        for cur_robot_id in range(self.num_robots):

            # roll dice to see if we can get a loop closure here. If greater
            # than this value then no loop closure
            if np.random.rand() > self.sim_params.loop_closure_prob or (len(self._groundtruth_poses[cur_robot_id]) - 1 in self._sim_params.no_loop_pose_idx):
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
                candidate_pose_chain = self._groundtruth_poses[loop_clos_robot_id][:-self.sim_params.exclude_last_n_poses_for_loop_closure]
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
                randomly_selected_pose = choice(possible_loop_closures)
                loop_closure = cur_robot.get_loop_closure_measurement(
                    randomly_selected_pose,
                    self.sim_params.groundtruth_measurements
                )
                self._loop_closures.append(loop_closure)

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
        assert isinstance(robot_1_idx, int)
        assert 0 <= robot_1_idx < self.num_robots
        assert isinstance(robot_2_idx, int)
        assert 0 <= robot_2_idx < self.num_robots
        assert robot_1_idx != robot_2_idx

        # first robot will always be correct?
        assoc_1 = self._robots[robot_1_idx].name

        # get all other robots
        true_other_assoc = self._robots[robot_2_idx].name
        robot_options = [x.name for x in self._robots]
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
        assert isinstance(robot_idx, int)
        assert 0 <= robot_idx < self.num_robots
        assert isinstance(beacon_idx, int)
        assert 0 <= beacon_idx < self.num_beacons

        # robot will always be correct?
        assoc_1 = self._robots[robot_idx].name

        # get all other robots
        robot_options = [x.name for x in self._robots]
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
        assert isinstance(robot_1_idx, int)
        assert 0 <= robot_1_idx < self.num_robots
        assert isinstance(robot_2_idx, int)
        assert 0 <= robot_2_idx < self.num_robots
        assert robot_1_idx < robot_2_idx

        assert isinstance(measurement, RangeMeasurement)
        assert 0.0 <= measurement.true_distance <= self.sim_params.range_sensing_radius

        self._range_measurements.append(measurement)

        # fill in the measurement info
        true_association = (
            self._robots[robot_1_idx].name,
            self._robots[robot_2_idx].name,
        )

        # randomly sample to decide if this is an incorrect association
        if np.random.rand() < self.sim_params.false_range_data_association_prob:
            measurement_association = (
                self._get_incorrect_robot_to_robot_range_association(
                    robot_1_idx, robot_2_idx
                )
            )
        else:
            measurement_association = true_association

        assert isinstance(measurement_association, tuple)
        assert all(isinstance(x, str) for x in measurement_association)

        self._range_associations.append(measurement_association)

        # fill in the groundtruth association
        self._groundtruth_range_associations.append(true_association)

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
        assert isinstance(robot_idx, int)
        assert 0 <= robot_idx < self.num_robots
        assert isinstance(beacon_idx, int)
        assert 0 <= beacon_idx < self.num_beacons

        assert isinstance(measurement, RangeMeasurement)

        # fill in the measurement info
        true_association = (
            self._robots[robot_idx].name,
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
        else:
            measurement_association = true_association

        assert isinstance(measurement_association, tuple)
        assert all(isinstance(x, str) for x in measurement_association)

        self._range_measurements.append(measurement)
        self._range_associations.append(measurement_association)

        # fill in the groundtruth info
        self._groundtruth_range_associations.append(true_association)

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
            assert isinstance(robot_plot_obj, matplotlib.lines.Line2D)
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
        self.fig.close()
