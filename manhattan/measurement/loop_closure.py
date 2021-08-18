import numpy as np

from manhattan.geometry.TwoDimension import SE2Pose
from numpy import ndarray


class LoopClosure:
    """
    represents a loop closure between poses
    """

    def __init__(
        self,
        pose_1: SE2Pose,
        pose_2: SE2Pose,
        measured_relative_pose: SE2Pose,
        timestamp: int,
        mean_offset: np.ndarray,
        covariance: np.ndarray,
    ) -> None:
        assert isinstance(pose_1, SE2Pose)
        assert isinstance(pose_2, SE2Pose)
        assert isinstance(measured_relative_pose, SE2Pose)
        assert isinstance(timestamp, int)
        assert isinstance(mean_offset, np.ndarray)
        assert mean_offset.shape == (3,)
        assert isinstance(covariance, np.ndarray)
        assert covariance.shape == (3, 3)

        self._pose_1 = pose_1
        self._pose_2 = pose_2
        self._measured_rel_pose = measured_relative_pose
        self._timestamp = timestamp
        self._true_transform = self._pose_1.transform_to(self._pose_2)
        self._mean_offset = mean_offset
        self._covariance = covariance

    @property
    def true_transformation(self) -> SE2Pose:
        """
        returns the transformation between the poses
        """
        return self._true_transform

    @property
    def measurement(self) -> SE2Pose:
        """
        returns the noisy transformation between the poses
        """
        return self._measured_rel_pose

    @property
    def timestamp(self) -> int:
        """
        returns the timestamp of the pose
        """
        return self._timestamp

    @property
    def mean_offset(self) -> np.ndarray:
        """
        returns the mean offset in the measurement model
        """
        return self._mean_offset

    @property
    def covariance(self) -> np.ndarray:
        """
        returns the covariance of the measurement
        """
        return self._covariance

