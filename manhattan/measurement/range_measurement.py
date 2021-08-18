class RangeMeasurement:
    """
    This class represents a range measurement.
    """

    def __init__(
        self,
        true_distance: float,
        measured_dist: float,
        mean: float,
        stddev: float,
        timestamp: int,
    ) -> None:
        """Constructs a range measurement

        Args:
            true_distance (float): the true distance
            measured_dist (float): the measured distance
            mean (float): the mean offset of the measurement model
            stddev (float): the standard deviation of the measurement model
            timestamp (int): the measurement timestamp
        """
        assert isinstance(true_distance, float)
        assert true_distance >= 0.0
        assert isinstance(measured_dist, float)
        assert measured_dist >= 0.0, f"measured_dist: {measured_dist},true_distance: {true_distance}"
        assert isinstance(mean, float)
        assert isinstance(stddev, float)
        assert stddev >= 0.0
        assert isinstance(timestamp, int)

        self._true_distance = true_distance
        self._measured_distance = measured_dist
        self._mean = mean
        self._stddev = stddev
        self._timestamp = timestamp

    def __str__(self):
        line = "Range Measurement\n"
        line += f"True distance: {self._true_distance}\n"
        line += f"Measured distance: {self._measured_distance}\n"
        line += f"Mean offset: {self._mean}\n"
        line += f"Stddev: {self._stddev}\n"
        line += f"Timestamp: {self._timestamp}\n"
        return line

    @property
    def true_distance(self) -> float:
        """
        The true distance
        """
        return self._true_distance

    @property
    def measured_distance(self) -> float:
        """
        The measured distance
        """
        return self._measured_distance

    @property
    def mean(self) -> float:
        """
        The mean
        """
        return self._mean

    @property
    def stddev(self) -> float:
        """
        The standard deviation
        """
        return self._stddev

    @property
    def timestamp(self) -> int:
        """
        The timestamp
        """
        return self._timestamp
