from typing import Tuple, Union
from abc import ABC, abstractmethod
from src.noise_models.range_noise_model import RangeNoiseModel
from src.geometry.TwoDimension import SE2Pose, Point2


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

    def get_range_measurement_to_agent(self, other_agent: Agent) -> float:
        other_loc = other_agent.get_groundtruth_position()
        assert isinstance(other_loc, Point2)
        cur_loc = self.get_groundtruth_position()
        assert isinstance(cur_loc, Point2)
        dist = cur_loc.distance(other_loc)
        measurement = self._range_model.get_range_measurement(dist)
        return measurement

    @abstractmethod
    def get_groundtruth_position(self) -> Point2:
        pass


class Robot(Agent):
    def __init__(
        self,
        name: str,
        start_pose: SE2Pose,
        range_model: RangeNoiseModel,
    ):
        super().__init__(name, range_model)

        assert isinstance(start_pose, SE2Pose)
        self._groundtruth_pose = start_pose

    def __str__(self) -> str:
        return "Robot: {}\n".format(self._name) + "Groundtruth pose: {}\n\n".format(
            self._groundtruth_pose
        )

    @property
    def get_groundtruth_position(self) -> Point2:
        assert isinstance(self._groundtruth_pose, SE2Pose)
        assert isinstance(self._groundtruth_pose.translation, Point2)
        return self._groundtruth_pose.get_translation()

    @property
    def get_groundtruth_pose(self) -> SE2Pose:
        assert isinstance(self._groundtruth_pose, SE2Pose)
        return self._groundtruth_pose


class Beacon(Agent):
    def __init__(
        self,
        name: str,
        start_position: Point2,
        range_model: RangeNoiseModel,
    ):
        super().__init__(name, range_model)
        self._groundtruth_position = start_position

    def __str__(self):
        return "Beacon: {}\n".format(
            self._name
        ) + "Groundtruth position: {}\n\n".format(self._groundtruth_position)

    @property
    def get_groundtruth_position(self) -> Point2:
        return self._groundtruth_position