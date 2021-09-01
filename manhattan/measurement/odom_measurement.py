from manhattan.geometry.TwoDimension import SE2Pose
import numpy as np
from numpy import ndarray


class OdomMeasurement:
    """
    This class represents an odometry measurement.
    """

    def __init__(
        self,
        true_odometry: SE2Pose,
        measured_odometry: SE2Pose,
        mean_offset: np.ndarray,
        covariance: np.ndarray,
    ) -> None:
        """Construct an odometry measurement

        Args:
            true_odometry (SE2Pose): the true odometry
            measured_odometry (SE2Pose): the measured odometry
        """
        assert isinstance(true_odometry, SE2Pose)
        assert isinstance(measured_odometry, SE2Pose)
        assert isinstance(mean_offset, np.ndarray)
        assert mean_offset.shape == (3,)
        assert isinstance(covariance, np.ndarray)
        assert covariance.shape == (3, 3)

        self._true_odom = true_odometry
        self._measured_odom = measured_odometry
        self._mean_offset = mean_offset
        self._covariance = covariance

    def __str__(self):
        line = "Odom Measurement\n"
        line += f"True Odometry: {self._true_odom}\n"
        line += f"Measured Odometry: {self._measured_odom}\n"
        line += f"Mean Offset: {self._mean_offset}\n"
        line += f"Covariance: {self._covariance.flatten()}"
        return line

    @property
    def local_frame(self) -> str:
        return self._measured_odom.local_frame

    @property
    def base_frame(self) -> str:
        return self._measured_odom.base_frame

    @property
    def true_odom(self) -> SE2Pose:
        """Get the true odometry"""
        return self._true_odom

    @property
    def measured_odom(self) -> SE2Pose:
        """Get the measured odometry"""
        return self._measured_odom

    @property
    def mean_offset(self) -> np.ndarray:
        """Get the mean offset"""
        return self._mean_offset

    @property
    def covariance(self) -> np.ndarray:
        """Get the covariance"""
        return self._covariance
