from manhattan.geometry.TwoDimension import SE2Pose


class OdomMeasurement:
    """
    This class represents an odometry measurement.
    """

    def __init__(
        self,
        true_odometry: SE2Pose,
        measured_odometry: SE2Pose,
    ):
        """Construct an odometry measurement

        Args:
            true_odometry (SE2Pose): the true odometry
            measured_odometry (SE2Pose): the measured odometry
        """
        self._true_odom = true_odometry
        self._measured_odom = measured_odometry

    @property
    def true_odom(self) -> SE2Pose:
        """Get the true odometry"""
        return self._true_odom

    @property
    def measured_odom(self) -> SE2Pose:
        """Get the measured odometry"""
        return self._measured_odom
