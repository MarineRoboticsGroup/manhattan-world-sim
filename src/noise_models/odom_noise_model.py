from abc import abstractmethod
import numpy as np

from src.noise_models.noise_model import NoiseModel
from src.measurement.odom_measurement import OdomMeasurement
from src.geometry.TwoDimension import SE2Pose


class OdomNoiseModel(NoiseModel):
    """
    A base odometry noise model.
    """

    def __init__(self, name: str):
        """
        Initialize this noise model.
        """
        assert isinstance(name, str)
        super().__init__(name)

    def __str__(self):
        return (
            f"Generic Odometry: {self._name}\n"
            + f"Covariance: {self._covariance}\n"
            + f"Mean: {self._mean}\n"
        )

    @abstractmethod
    def get_odometry_measurement(self, movement: SE2Pose) -> OdomMeasurement:
        """Get a noisy odometry measurement from the true odometry.

        Args:
            movement (SE2Pose): the movement performed by the robot

        Returns:
            OdomMeasurement: the noisy measurement of the movement
        """
        pass


class GaussianOdomNoiseModel(OdomNoiseModel):
    """
    This is a simple Gaussian noise model for the robot odometry. This assumes
    that the noise is additive gaussian and constant in time and distance moved.
    """

    def __init__(
        self,
        name: str,
        mean: np.ndarray = np.zeros(3),
        covariance: np.ndarray = np.eye(3),
    ):
        """Initializes the gaussian additive noise model

        Args:
            name (str): name for the noise model
            mean (np.ndarray, optional): The mean of the noise. Generally
                zero-mean. Entries are [x, y, theta]. Defaults to np.zeros(3).
            covariance (np.ndarray, optional): The covariance matrix
                with entries corresponding to the mean vector. Defaults to
                np.eye(3).
        """
        assert isinstance(name, str)
        assert isinstance(covariance, np.ndarray)
        assert covariance.shape == (3, 3)
        assert isinstance(mean, np.ndarray)
        assert mean.shape == (3,)
        super().__init__(name)
        self._mean = mean
        self._covariance = covariance

    def __str__(self):
        return (
            f"Gaussian Additive Odometry: {self._name}\n"
            + f"Covariance: {self._covariance}\n"
            + f"Mean: {self._mean}\n"
        )

    @property
    def covariance(self):
        return self._covariance

    @property
    def mean(self):
        return self._mean

    def get_odometry_measurement(self, movement: SE2Pose) -> OdomMeasurement:
        """Takes the groundtruth movement performed and then perturbs it by
        transformation randomly sampled from a Gaussian distribution and passed
        through the exponential map

        Args:
            movement (SE2Pose): the true movement performed by the robot

        Returns:
            SE2Pose: A noisy measurement of the movement passed in
        """
        assert isinstance(movement, SE2Pose)

        # this is the constant component from the gaussian noise
        mean_offset = SE2Pose.by_exp_map(self._mean)

        # this is the random component from the gaussian noise
        noise_sample = np.random.multivariate_normal(np.zeros(3), self._covariance)
        noise_offset = SE2Pose.by_exp_map(noise_sample)

        # because we're in 2D rotations commute so we don't need to think about
        # the order of operations
        noisy_odom_measurement = movement * mean_offset * noise_offset
        return OdomMeasurement(movement, noisy_odom_measurement)
