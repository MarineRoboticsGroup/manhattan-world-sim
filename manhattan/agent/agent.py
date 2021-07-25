from __future__ import annotations
from abc import abstractmethod
import matplotlib.pyplot as plt

from manhattan.noise_models.range_noise_model import RangeNoiseModel
from manhattan.noise_models.odom_noise_model import OdomNoiseModel
from manhattan.measurement.range_measurement import RangeMeasurement
from manhattan.measurement.odom_measurement import OdomMeasurement
from manhattan.geometry.TwoDimension import SE2Pose, Point2


# TODO need to make classes hashable!
class Agent:
    """
    This class represents a general agent. In our simulator this is either a
    robot or a stationary beacon.
    """

    def __init__(
        self,
        name: str,
        range_model: RangeNoiseModel,
    ):
        assert isinstance(name, str)
        assert isinstance(range_model, RangeNoiseModel)

        self._name = name
        self._range_model = range_model

    def get_range_measurement_to_agent(self, other_agent: Agent) -> RangeMeasurement:
        other_loc = other_agent.get_groundtruth_position()
        assert isinstance(other_loc, Point2)
        cur_loc = self.get_groundtruth_position()
        assert isinstance(cur_loc, Point2)
        dist = cur_loc.distance(other_loc)
        measurement = self._range_model.get_range_measurement(dist)
        assert isinstance(measurement, RangeMeasurement)
        return measurement

    @abstractmethod
    def get_groundtruth_position(self) -> Point2:
        pass

    @abstractmethod
    def plot(self) -> None:
        pass


class Robot(Agent):
    def __init__(
        self,
        name: str,
        start_pose: SE2Pose,
        range_model: RangeNoiseModel,
        odometry_model: OdomNoiseModel,
    ):
        assert isinstance(name, str)
        assert isinstance(start_pose, SE2Pose)
        assert isinstance(range_model, RangeNoiseModel)
        assert isinstance(odometry_model, OdomNoiseModel)

        super().__init__(name, range_model)
        self._odom_model = odometry_model
        self._groundtruth_pose = start_pose

    def __str__(self) -> str:
        return (
            f"Robot: {self._name}\n"
            + f"Groundtruth pose: {self._groundtruth_pose}\n"
            + f"Range model: {self._range_model}\n"
            + f"Odometry model: {self._odom_model}\n"
        )

    @property
    def groundtruth_position(self) -> Point2:
        assert isinstance(self._groundtruth_pose, SE2Pose)
        assert isinstance(self._groundtruth_pose.translation, Point2)
        return self._groundtruth_pose.translation

    @property
    def groundtruth_pose(self) -> SE2Pose:
        assert isinstance(self._groundtruth_pose, SE2Pose)
        return self._groundtruth_pose

    def move(self, transform: SE2Pose) -> OdomMeasurement:
        """Moves the robot by the given transform and returns a noisy odometry
        measurement of the move

        Args:
            transform (SE2Pose): the transform to move the robot by

        Returns:
            SE2Pose: the noisy measurement of the transform
        """
        assert isinstance(transform, SE2Pose)

        # move the robot
        self._groundtruth_pose = self._groundtruth_pose * transform

        # get the odometry measurement
        odom_measurement = self._odom_model.get_odometry_measurement(transform)
        assert isinstance(odom_measurement, OdomMeasurement)
        return odom_measurement

    def plot(self) -> None:
        """Plots the robot's groundtruth position"""
        cur_position = self.groundtruth_position
        plt.plot(cur_position.x, cur_position.y, "bx", markersize=10)



class Beacon(Agent):
    def __init__(
        self,
        name: str,
        position: Point2,
        range_model: RangeNoiseModel,
    ):
        assert isinstance(name, str)
        assert isinstance(position, Point2)
        assert isinstance(range_model, RangeNoiseModel)

        super().__init__(name, range_model)
        self._groundtruth_position = position

    def __str__(self):
        return (
            f"Beacon: {self._name}\n"
            + f"Groundtruth position: {self._groundtruth_position}\n"
            + f"Range model: {self._range_model}\n"
        )

    @property
    def groundtruth_position(self) -> Point2:
        return self._groundtruth_position

    def plot(self) -> None:
        """Plots the beacons's groundtruth position"""
        cur_position = self.groundtruth_position
        plt.plot(cur_position.x, cur_position.y, "g+", markersize=10)