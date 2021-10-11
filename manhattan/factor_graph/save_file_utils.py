# from .nf_isam.example.slam.manhattan_waterworld.factor_graph_generator import
# *
from typing import List, Tuple
import re
import numpy as np

# from manhattan.agent.agent import Robot, Beacon
# from manhattan.geometry.TwoDimension import SE2Pose, Point2
# from manhattan.measurement.range_measurement import RangeMeasurement
# from manhattan.measurement.odom_measurement import OdomMeasurement
# from manhattan.measurement.loop_closure import LoopClosure
from manhattan.factor_graph.factor_graph import (
    FactorGraphData,
    PoseVariable,
    LandmarkVariable,
    OdomMeasurement,
    AmbiguousOdomMeasurement,
    RangeMeasurement,
    AmbiguousRangeMeasurement,
    PosePrior,
    LandmarkPrior,
)
from manhattan.factor_graph.name_utils import (
    get_robot_char_from_frame_name,
    get_idx_from_frame_name,
    check_is_valid_frame_name,
)


def save_to_efg_format(
    data_file: str,
    factor_graph: FactorGraphData
    # odom_measurements: List[List[OdomMeasurement]],
    # loop_closures: List[LoopClosure],
    # gt_poses: List[List[SE2Pose]],
    # beacons: List[Beacon],
    # range_measurements: List[RangeMeasurement],
    # range_associations: List[Tuple[str, str]],
    # gt_range_associations: List[Tuple[str, str]],
) -> None:
    """
    Save the given data to the extended factor graph format.
    """

    def get_normal_pose_measurement_string(pose_measure: OdomMeasurement) -> str:
        """This is a utility function to get a formatted string to write to EFG
        formats for measurements which can be represented by poses (i.e.
        odometry and loop closures.

        Args:
            pose (OdomMeasurement): the measurement

        Returns:
            str: the formatted string representation of the pose measurement
        """
        # add in odometry info
        del_x = pose_measure.x
        del_y = pose_measure.y
        del_theta = pose_measure.theta
        line = f"{del_x:.15f} {del_y:.15f} {del_theta:.15f} "

        # add in covariance info
        line += "covariance "
        covar_info = pose_measure.covariance.flatten()
        for val in covar_info:
            line += f"{val:.15f} "

        # return the formatted string
        return line

    def get_ambiguous_pose_measurement_string(
        pose: AmbiguousOdomMeasurement, cov: np.ndarray
    ) -> str:
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
        line = f"{del_x:.15f} {del_y:.15f} {del_theta:.15f} "

        # add in covariance info (Sigma == covariance)
        line += "Sigma "
        covar_info = cov.flatten()
        for val in covar_info:
            line += f"{val:.15f} "

        # return the formatted string
        return line

    def get_pose_var_string(pose: PoseVariable) -> str:
        """
        Takes a pose and returns a string in the desired format
        """
        line = "Variable Pose SE2 "

        # get local frame for pose
        pose_key = pose.name

        # add in pose information
        line += (
            f"{pose_key} {pose.true_x:.15f} {pose.true_y:.15f} {pose.true_theta:.15f}\n"
        )

        return line

    def get_beacon_var_string(beacon: LandmarkVariable) -> str:
        """Takes in a beacon and returns a string formatted as desired

        Args:
            beacon (Beacon): the beacon

        Returns:
            str: the formatted string
        """
        line = "Variable Landmark R2 "

        frame = beacon.name
        line += f"{frame} {beacon.true_x:.15f} {beacon.true_y:.15f}\n"
        return line

    def get_prior_to_pin_string(prior: PosePrior) -> str:
        """this is the prior on the first pose to 'pin' the factor graph.

        Returns:
            str: the line representing the prior
        """
        prior_key = prior.name
        line = f"Factor UnarySE2ApproximateGaussianPriorFactor {prior_key} "
        line += f"{prior.x:.15f} {prior.y:.15f} {prior.theta:.15f} "

        line += "covariance "
        cov = prior.covariance.flatten()
        for val in cov:
            line += f"{val:.15f} "
        line += "\n"

        return line

    def get_range_measurement_string(
        range_measure: RangeMeasurement,
    ) -> str:
        """Returns the string representing a range factor based on the provided
        range measurement and the association information.

        Args:
            range_measure (RangeMeasurement): the measurement info (value and
                stddev)

        Returns:
            str: the line representing the factor
        """

        robot_id, measure_id = range_measure.association

        # Factor SE2R2RangeGaussianLikelihoodFactor X0 L1 14.14214292904807 0.5
        if "L" in measure_id:
            # L is reserved for landmark names
            range_factor_type = "SE2R2RangeGaussianLikelihoodFactor"
        else:
            # ID starts with other letters are robots
            range_factor_type = "SE2SE2RangeGaussianLikelihoodFactor"
        line = f"Factor {range_factor_type} "
        line += f"{robot_id} {measure_id} "
        line += f"{range_measure.dist:.15f} {range_measure.stddev:.15f}\n"

        return line

    def get_ambiguous_range_measurement_string(
        range_measure: AmbiguousRangeMeasurement,
    ) -> str:
        """Returns the string representing an ambiguous range factor based on
        the provided range measurement and the association information.

        Args:
            range_measure (AmbiguousRangeMeasurement): the measurement info
                (value and stddev)

        Returns:
            str: the line representing the factor
        """

        true_robot_id, true_beacon_id = range_measure.true_association
        measured_robot_id, measured_beacon_id = range_measure.measured_association

        assert true_robot_id == measured_robot_id, "the robot id must always be correct"
        assert (
            "L" in true_beacon_id and "L" in measured_beacon_id
        ), "right now only considering ambiguous measurements to landmarks"

        # Factor AmbiguousDataAssociationFactor Observer X1 Observed L1 L2
        # Weights 0.5 0.5 Binary SE2R2RangeGaussianLikelihoodFactor
        # Observation 24.494897460297107 Sigma 0.5
        line = "Factor AmbiguousDataAssociationFactor "
        line += f"Observer {true_robot_id} "
        line += f"Observed {true_beacon_id} {measured_beacon_id} "
        line += "Weights 0.5 0.5 Binary SE2R2RangeGaussianLikelihoodFactor "
        line += (
            f"Observation {range_measure.dist:.15f} Sigma {range_measure.stddev:.15f}\n"
        )

        return line

    file_writer = open(data_file, "w")

    for pose in factor_graph.pose_variables:
        line = get_pose_var_string(pose)
        file_writer.write(line)

    for beacon in factor_graph.landmark_variables:
        line = get_beacon_var_string(beacon)
        file_writer.write(line)

    for prior in factor_graph.pose_priors:
        line = get_prior_to_pin_string(prior)
        file_writer.write(line)

    for odom_measure in factor_graph.odom_measurements:
        line = get_normal_pose_measurement_string(odom_measure)
        file_writer.write(line)

    for range_measure in factor_graph.range_measurements:
        line = get_range_measurement_string(range_measure)
        file_writer.write(line)

    file_writer.close()
