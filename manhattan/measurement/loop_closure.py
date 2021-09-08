import numpy as np
import attr

from manhattan.geometry.TwoDimension import SE2Pose
from numpy import ndarray


@attr.s(frozen=True)
class LoopClosure:
    """
    represents a loop closure between poses

    pose_1 (SE2Pose): the pose the loop closure is measured from
    pose_2 (SE2Pose): the pose the loop closure is measured to
    measured_association (str): the measurement association (can be
        incorrect and differ from the true association)
    measured_relative_pose (SE2Pose): the measured relative pose
    timestamp (int): the timestamp of the measurement
    mean_offset (np.ndarray): the mean offset in the measurement model
    covariance (np.ndarray): the covariance of the measurement model
    """

    pose_1: SE2Pose = attr.ib()
    pose_2: SE2Pose = attr.ib()
    measured_association: str = attr.ib()
    measured_rel_pose: SE2Pose = attr.ib()
    timestamp: int = attr.ib()
    mean_offset: np.ndarray = attr.ib()
    covariance: np.ndarray = attr.ib()

    def __str__(self) -> str:
        return (
            f"LoopClosure (t={self.timestamp})\n"
            f"{self.pose_1} -> {self.pose_2}\n"
            f"measured association: {self.measured_association}\n"
            f"{self.measured_rel_pose}\n"
            f"offset: {self.mean_offset}\n"
            f"covariance:\n{self.covariance}"
        )

    @property
    def true_association(self) -> str:
        """
        the true association between poses
        """
        return self.pose_2.local_frame

    @property
    def true_transformation(self):
        return self.pose_1.transform_to(self.pose_2)

    @property
    def base_frame(self) -> str:
        return self.pose_1.local_frame

    @property
    def local_frame(self) -> str:
        return self.pose_2.local_frame

    @property
    def measurement(self) -> SE2Pose:
        """
        returns the noisy transformation between the poses
        """
        return self.measured_rel_pose
