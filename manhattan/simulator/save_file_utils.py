# from .nf_isam.example.slam.manhattan_waterworld.factor_graph_generator import
# *
from typing import List, Tuple
import re
import numpy as np

from manhattan.agent.agent import Robot, Beacon
from manhattan.geometry.TwoDimension import SE2Pose, Point2
from manhattan.measurement.range_measurement import RangeMeasurement
from manhattan.measurement.odom_measurement import OdomMeasurement
from manhattan.measurement.loop_closure import LoopClosure


def _get_robot_char_from_number(robot_number: int) -> str:
    """
    Get the robot character from the given robot number.
    """
    char = chr(ord("A") + robot_number)
    assert char != "L", "Character L is reserved for landmarks"
    return "X"
    return char


def get_robot_char_from_frame_name(frame: str) -> str:
    """
    Get the robot character from the given frame.
    """
    assert "Robot " in frame
    robot_name = re.search(r"Robot [\d+]*", frame).group(0)  # type: ignore
    robot_number = int(robot_name[len("Robot ") :])
    return _get_robot_char_from_number(robot_number)


def get_pose_key_from_frame_name(frame: str) -> str:
    """
    Get the pose key from the given frame.
    """
    assert "Robot " in frame

    # get the char corresponding to the robot (e.g. A for robot 0, B for
    # robot 1, etc.)
    key_char = get_robot_char_from_frame_name(frame)

    # get the timestamp for the pose
    time_str = re.search(r"time: [\d+]*", frame).group(0)  # type: ignore
    pose_idx = int(time_str[len("time: ") :])
    return f"{key_char}{pose_idx}"


def save_to_efg_format(
    data_file: str,
    odom_measurements: List[List[OdomMeasurement]],
    loop_closures: List[LoopClosure],
    gt_poses: List[List[SE2Pose]],
    beacons: List[Beacon],
    range_measurements: List[RangeMeasurement],
    range_associations: List[Tuple[str, str]],
    gt_range_associations: List[Tuple[str, str]],
) -> None:
    """
    Save the given data to the extended factor graph format.
    """

    def get_pose_measurement_string(pose: SE2Pose, cov: np.ndarray) -> str:
        """This is a utility function to get a formatted string to write to EFG
        formats for measurements which can be represented by poses (i.e.
        odometry and loop closures.

        Args:
            pose (SE2Pose): the measurement

        Returns:
            str: the formatted string representation of the pose measurement
        """
        assert cov.shape == (3, 3)  # because only in SE2 (3 variables)

        # add in odometry info
        del_x = pose.x
        del_y = pose.y
        del_theta = pose.theta
        line = f"{del_x} {del_y} {del_theta} "

        # add in covariance info
        line += "covariance "
        covar_info = cov.flatten()
        for val in covar_info:
            line += f"{val:.15f} "

        # return the formatted string
        return line

    def get_pose_var_string(pose: SE2Pose) -> str:
        """
        Takes a pose and returns a string in the desired format
        """
        line = "Variable Pose SE2 "

        # get local frame for pose
        frame = pose.local_frame

        # get the key to associate with this pose
        pose_key = get_pose_key_from_frame_name(frame)

        # add in pose information
        line += f"{pose_key} {pose.x:.15f} {pose.y:.15f} {pose.theta:.15f}\n"

        return line

    def get_beacon_var_string(beacon: Beacon) -> str:
        """Takes in a beacon and returns a string formatted as desired

        Args:
            beacon (Beacon): the beacon

        Returns:
            str: the formatted string
        """
        line = "Variable Landmark R2 "

        frame = beacon.name
        search_str = "Beacon "
        beacon_str_ind = frame.find(search_str)
        beacon_ind = int(frame[beacon_str_ind + len(search_str) :])
        pos = beacon.position
        line += f"L{beacon_ind} {pos.x:.15f} {pos.y:.15f}\n"
        return line

    def get_prior_to_pin_string(poses: List[List[SE2Pose]]) -> str:
        """this is the prior on the first pose to 'pin' the factor graph.

        Returns:
            str: the line representing the prior
        """
        pinned_pose = poses[0][0]
        pose_key = get_pose_key_from_frame_name(pinned_pose.local_frame)
        line = f"Factor UnarySE2ApproximateGaussianPriorFactor {pose_key} "
        line += f"{pinned_pose.x:.15f} {pinned_pose.y:.15f} {pinned_pose.theta:.15f} "
        x_stddev = 1
        y_stddev = 1
        theta_stddev = 1 / 20
        line += f"covariance {x_stddev**2} 0.0 0.0 0.0 {y_stddev**2} 0.0 0.0 0.0 {theta_stddev**2}\n"
        return line

    def get_odom_factor_string(odom_measurement: OdomMeasurement) -> str:
        """
        Takes in an odometry measurement and returns a string in the desired
        format for a factor.
        """
        # Factor SE2RelativeGaussianLikelihoodFactor X0 X1 20 0
        # 0.7853981633974483 covariance 0.010000000000000002 0.0 0.0 0.0
        # 0.010000000000000002 0.0 0.0 0.0 0.0009
        # set up basic information
        line = "Factor SE2RelativeGaussianLikelihoodFactor "
        base_key = get_pose_key_from_frame_name(odom_measurement.base_frame)
        to_key = get_pose_key_from_frame_name(odom_measurement.local_frame)

        line += f"{base_key} {to_key} "

        odom_info = get_pose_measurement_string(
            odom_measurement.measured_odom, odom_measurement.covariance
        )
        line += odom_info

        # covar_info = covar_info.replace("[", "").replace("]", "")
        line += "\n"
        return line

    def get_loop_closure_factor_string(loop_clos: LoopClosure) -> str:
        """
        Takes in a loop closure and returns a string in the desired format
        for a factor.

        Args:
            loop_clos (LoopClosure): the loop closure

        Returns:
            str: the line representing the factor
        """

        # Factor SE2RelativeGaussianLikelihoodFactor B2 B3 0.9102598585313532
        # 0.06585288343000831 -0.0015874335137571194 covariance
        # 0.010000000000000 0.000000000000000 0.000000000000000
        # 0.000000000000000 0.010000000000000 0.000000000000000
        # 0.000000000000000 0.000000000000000 0.000100000000000
        line = "Factor SE2RelativeGaussianLikelihoodFactor "

        # add in which poses are being related
        base_pose_key = get_pose_key_from_frame_name(loop_clos.base_frame)
        to_pose_key = get_pose_key_from_frame_name(loop_clos.local_frame)
        line += f"{base_pose_key} {to_pose_key} "

        # add in the measurement
        loop_clos_data = get_pose_measurement_string(
            loop_clos.measurement, loop_clos.covariance
        )
        line += loop_clos_data

        line += "\n"
        return line

    def get_range_factor_string(
        range_measure: RangeMeasurement,
        association: Tuple[str, str],
        true_association: Tuple[str, str],
    ) -> str:
        def get_association_variable_str(association_var: str, timestamp: int) -> str:
            assert isinstance(association_var, str)
            assert "Robot" in association_var or "Beacon" in association_var
            assert isinstance(timestamp, int)
            assert 0 <= timestamp

            if "Robot" in association_var:
                robot_char = get_robot_char_from_frame_name(association_var)
                return f"{robot_char}{timestamp}"
            elif "Beacon" in association_var:
                beacon_str = "Beacon "
                beacon_idx = int(association_var[len(beacon_str) :])
                return f"L{beacon_idx}"

            raise ValueError(
                f"association_var: {association_var},timestamp: {timestamp}"
            )

        timestamp = range_measure.timestamp

        assert association[0] != association[1]
        assert true_association[0] != true_association[1]

        robot_str = association[0]
        assert robot_str == true_association[0]

        # get the key of the pose (e.g. A1) for Robot 0, timestep 1
        robot_id = get_association_variable_str(robot_str, timestamp)

        # get the key of the pose (e.g. A1) for Robot 0, timestep 1
        measure_id = association[1]
        measure_id = get_association_variable_str(measure_id, timestamp)
        true_measure_id = true_association[1]
        true_measure_id = get_association_variable_str(true_measure_id, timestamp)

        if measure_id == true_measure_id:
            # Factor SE2R2RangeGaussianLikelihoodFactor X0 L1 14.14214292904807 0.5
            line = "Factor SE2R2RangeGaussianLikelihoodFactor "
            line += f"{robot_id} {true_measure_id} "
            line += (
                f"{range_measure.measured_distance:.15f} {range_measure.stddev:.15f}\n"
            )
        else:
            # Factor AmbiguousDataAssociationFactor Observer X1 Observed L1 L2
            # Weights 0.5 0.5 Binary SE2R2RangeGaussianLikelihoodFactor
            # Observation 24.494897460297107 Sigma 0.5
            line = "Factor AmbiguousDataAssociationFactor "
            line += f"Observer {robot_id} "
            line += f"Observed {true_measure_id} {measure_id} "
            line += "Weights 0.5 0.5 Binary SE2R2RangeGaussianLikelihoodFactor "
            line += f"Observation {range_measure.measured_distance:.15f} Sigma {range_measure.stddev:.15f}\n"

        return line

    file_writer = open(data_file, "w")

    for pose_chain in gt_poses:
        for pose in pose_chain:
            line = get_pose_var_string(pose)
            file_writer.write(line)

    for beacon in beacons:
        line = get_beacon_var_string(beacon)
        file_writer.write(line)

    line = get_prior_to_pin_string(gt_poses)
    file_writer.write(line)

    for odom_measure_chain in odom_measurements:
        for odom_measure in odom_measure_chain:
            line = get_odom_factor_string(odom_measure)
            file_writer.write(line)

    for loop_closure in loop_closures:
        line = get_loop_closure_factor_string(loop_closure)
        file_writer.write(line)

    for range_idx in range(len(range_measurements)):
        line = get_range_factor_string(
            range_measurements[range_idx],
            range_associations[range_idx],
            gt_range_associations[range_idx],
        )
        file_writer.write(line)

    file_writer.close()
