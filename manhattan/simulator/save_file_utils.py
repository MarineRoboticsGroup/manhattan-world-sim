# from .nf_isam.example.slam.manhattan_waterworld.factor_graph_generator import
# *
from typing import List, Tuple

from manhattan.agent.agent import Robot, Beacon
from manhattan.geometry.TwoDimension import SE2Pose, Point2
from manhattan.measurement.range_measurement import RangeMeasurement
from manhattan.measurement.odom_measurement import OdomMeasurement


def save_to_chad_format(
    data_file: str,
    odom_measurements: List[List[OdomMeasurement]],
    gt_poses: List[List[SE2Pose]],
    beacons: List[Beacon],
    range_measurements: List[RangeMeasurement],
    range_associations: List[Tuple[str, str]],
    gt_range_associations: List[Tuple[str, str]],
):
    """
    Save the given data to the file format used by CHAD.
    """

    def get_pose_string(pose: SE2Pose) -> str:
        """
        Takes a pose and returns a string in the desired format
        """
        line = "Variable Pose SE2 "

        # get timestamp for pose
        frame = pose.local_frame
        search_str = "time: "
        time_ind = frame.find(search_str)
        time = int(frame[time_ind + len(search_str) :])

        # add in pose information
        line += f"X{time} {pose.x:.3f} {pose.y:.3f} {pose.theta:.3f}\n"

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
        line += f"L{beacon_ind} {pos.x:.3f} {pos.y:.3f}\n"
        return line

    def get_prior_to_pin() -> str:
        """this is the prior on the first pose to 'pin' the factor graph.

        Raises:
            NotImplementedError: [description]

        Returns:
            str: [description]
        """
        line = "Factor UnarySE2ApproximateGaussianPriorFactor "
        line += "X0 0.0 0.0 0.0 "
        line += "covariance 0.01 0.0 0.0 0.0 0.01 0.0 0.0 0.0 0.001\n"
        return line

    def get_odom_factor_string(odom_measurement: OdomMeasurement, odom_num: int) -> str:
        """
        Takes in an odometry measurement and returns a string in the desired
        format for a factor.
        """
        # Factor SE2RelativeGaussianLikelihoodFactor X0 X1 20 0
        # 0.7853981633974483 covariance 0.010000000000000002 0.0 0.0 0.0
        # 0.010000000000000002 0.0 0.0 0.0 0.0009
        # set up basic information
        line = "Factor SE2RelativeGaussianLikelihoodFactor "
        line += f"X{odom_num} X{odom_num + 1} "

        # add in odometry info
        del_x = odom_measurement.measured_odom.x
        del_y = odom_measurement.measured_odom.y
        del_theta = odom_measurement.measured_odom.theta
        line += f"{del_x} {del_y} {del_theta} "

        # add in covariance info
        line += "covariance "
        covar_info = odom_measurement.covariance.flatten()
        for val in covar_info:
            line += f"{val:.3f} "

        # covar_info = covar_info.replace("[", "").replace("]", "")
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
                print(
                    "We currently aren't formatting data to account for multiple robots!",
                    "There should only be one robot in the simulator!!",
                )
                assert (
                    False
                ), "This shouldn't be triggered right now as Chad doesn't want to consider multiple robots"
                rob_str = "Robot "
                robot_idx = int(association_var[len(rob_str) :])
                return f"X{timestamp}"
            elif "Beacon" in association_var:
                beacon_str = "Beacon "
                beacon_idx = int(association_var[len(beacon_str) :])
                return f"L{beacon_idx}"

        timestamp = range_measure.timestamp

        assert association[0] != association[1]
        assert true_association[0] != true_association[1]

        robot_id = association[0]
        assert robot_id == true_association[0]
        assert robot_id == "Robot 0"

        robot_id = f"X{timestamp}"

        measure_id = association[1]
        measure_id = get_association_variable_str(measure_id, timestamp)
        true_measure_id = true_association[1]
        true_measure_id = get_association_variable_str(true_measure_id, timestamp)

        if measure_id == true_measure_id:
            # Factor SE2R2RangeGaussianLikelihoodFactor X0 L1 14.14214292904807 0.5
            line = "Factor SE2R2RangeGaussianLikelihoodFactor "
            line += f"{robot_id} {true_measure_id} "
            line += (
                f"{range_measure.measured_distance:.3f} {range_measure.stddev:.3f}\n"
            )
        else:
            # Factor AmbiguousDataAssociationFactor Observer X1 Observed L1 L2
            # Weights 0.5 0.5 Binary SE2R2RangeGaussianLikelihoodFactor
            # Observation 24.494897460297107 Sigma 0.5
            line = "Factor AmbiguousDataAssociationFactor "
            line += f"Observer {robot_id} "
            line += f"Observed {true_measure_id} {measure_id} "
            line += "Weights 0.5 0.5 Binary SE2R2RangeGaussianLikelihoodFactor "
            line += f"Observation {range_measure.measured_distance:.3f} Sigma {range_measure.stddev:.3f}\n"

        return line

    file_writer = open(data_file, "w")

    for pose_chain in gt_poses:
        for pose in pose_chain:
            line = get_pose_string(pose)
            file_writer.write(line)

    for beacon in beacons:
        line = get_beacon_var_string(beacon)
        file_writer.write(line)

    line = get_prior_to_pin()
    file_writer.write(line)

    for odom_measure_chain in odom_measurements:
        for odom_num, odom_measure in enumerate(odom_measure_chain):
            line = get_odom_factor_string(odom_measure, odom_num)
            file_writer.write(line)

    for range_idx in range(len(range_measurements)):
        line = get_range_factor_string(
            range_measurements[range_idx],
            range_associations[range_idx],
            gt_range_associations[range_idx],
        )
        file_writer.write(line)

    file_writer.close()
