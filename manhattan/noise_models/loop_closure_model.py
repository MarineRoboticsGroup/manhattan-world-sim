from abc import abstractmethod
import numpy as np

from manhattan.measurement.loop_closure import LoopClosure
from manhattan.geometry.TwoDimension import SE2Pose
from numpy import ndarray


class LoopClosureModel:
    """
    A base noisy loop closure model.
    """

    def __init__(
        self,
    ):
        """
        Initialize this noise model.
        """
        self._covariance = None
        self._mean = None

    def __str__(self):
        return (
            f"Generic Loop Closure Model\n"
            + f"Covariance: {self._covariance.flatten()}\n"
            + f"Mean: {self._mean}\n"
        )

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @abstractmethod
    def get_relative_pose_measurement(
        self, pose_1: SE2Pose, pose_2: SE2Pose, association: str, timestamp: int
    ) -> LoopClosure:
        """Takes a two poses, and returns a loop closure measurement based on
        the relative pose from pose_1 to pose_2 and the determined sensor model.

        Note: we get a little hacky with the frame naming just to allow for
        multiplication of all of these noises

        Args:
            pose_1 (SE2Pose): the first pose
            pose_2 (SE2Pose): the second pose
            association (str): the measured association of the second pose
            timestamp (int): the timestamp of the measurement

        Returns:
            LoopClosure: A noisy measurement of the relative pose from pose_1 to
                pose_2
        """
        pass


class GaussianLoopClosureModel(LoopClosureModel):
    """
    This is a simple Gaussian noise model for the robot odometry. This assumes
    that the noise is additive gaussian and constant in time and distance moved.
    """

    def __init__(
        self, mean: np.ndarray = np.zeros(3), covariance: np.ndarray = np.eye(3) / 50.0
    ) -> None:
        """Initializes the gaussian additive noise model

        Args:
            mean (np.ndarray, optional): The mean of the noise. Generally
                zero-mean. Entries are [x, y, theta]. Defaults to np.zeros(3).
            covariance (np.ndarray, optional): The covariance matrix
                with entries corresponding to the mean vector. Defaults to
                np.eye(3)/10.0.
        """
        assert isinstance(mean, np.ndarray)
        assert mean.shape == (3,)
        assert isinstance(covariance, np.ndarray)
        assert covariance.shape == (3, 3)
        self._mean = mean
        self._covariance = covariance

    def __str__(self):
        return (
            f"Gaussian Additive Loop Closure Model\n"
            + f"Covariance: {self._covariance.flatten()}\n"
            + f"Mean: {self._mean}\n"
        )

    def get_relative_pose_measurement(
        self, pose_1: SE2Pose, pose_2: SE2Pose, association: str, timestamp: int
    ) -> LoopClosure:
        """Takes a two poses, gets the relative pose and then perturbs it by
        transformation randomly sampled from a Gaussian distribution and passed
        through the exponential map

        Note: we get a little hacky with the frame naming just to allow for
        multiplication of all of these noises

        Args:
            pose_1 (SE2Pose): the first pose
            pose_2 (SE2Pose): the second pose
            association (str): the measured association of the second pose
            timestamp (int): the timestamp of the measurement

        Returns:
            SE2Pose: A noisy measurement of the relative pose from pose_1 to
                pose_2
        """
        assert isinstance(pose_1, SE2Pose)
        assert isinstance(pose_2, SE2Pose)
        assert isinstance(timestamp, int)
        assert 0 <= timestamp

        rel_pose = pose_1.transform_to(pose_2)

        # this is the constant component from the gaussian noise
        mean_offset = SE2Pose.by_exp_map(
            self._mean, local_frame="temp", base_frame=rel_pose.local_frame
        )

        # this is the random component from the gaussian noise
        noise_sample = np.random.multivariate_normal(np.zeros(3), self._covariance)
        noise_offset = SE2Pose.by_exp_map(
            noise_sample, local_frame=rel_pose.local_frame, base_frame="temp"
        )

        # because we're in 2D rotations commute so we don't need to think about
        # the order of operations???
        noisy_pose_measurement = rel_pose * mean_offset * noise_offset
        return LoopClosure(
            pose_1,
            pose_2,
            association,
            noisy_pose_measurement,
            timestamp,
            self._mean,
            self._covariance,
        )
