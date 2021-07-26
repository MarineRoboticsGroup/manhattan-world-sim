class RangeMeasurement:
    """
    This class represents a range measurement.
    """

    def __init__(self, true_distance: float, measured_dist: float):
        """Constructs a range measurement

        Args:
            true_distance (float): the true distance
            measured_dist (float): the measured distance
        """
        self._true_distance = true_distance
        self._measured_distance = measured_dist

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