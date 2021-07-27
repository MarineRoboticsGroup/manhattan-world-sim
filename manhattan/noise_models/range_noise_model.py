import numpy as np
from typing import Callable
from types import FunctionType

from manhattan.measurement.range_measurement import RangeMeasurement


class RangeNoiseModel:
    """
    Base range noise model. Takes in a function as an argument which describes
    the sensor model as follows:

    measurement = f(true_distance, sensor_params)
    """

    def __init__(self, measurement_model: Callable[[float], float]):
        assert isinstance(measurement_model, FunctionType)
        self._measurement_model = measurement_model

    def __str__(self):
        return (
            f"Generic Range Model\n"
            + f"Mean: {self._mean}\n"
            + f"Std Dev: {self._stddev}\n"
        )

    @property
    def measurement_model(self) -> Callable[[float, int], RangeMeasurement]:
        """This returns a function that takes in a distance and returns a measurement.

        Returns:
            Callable[[float, int],RangeMeasurement]: the measurement model function
        """
        return self._measurement_model

    def get_range_measurement(
        self, true_dist: float, timestamp: int
    ) -> RangeMeasurement:
        assert isinstance(true_dist, float)
        assert 0 <= true_dist
        assert isinstance(timestamp, int)
        assert 0 <= timestamp

        measurement = self._measurement_model(true_dist, timestamp)
        assert isinstance(
            measurement, RangeMeasurement
        ), f"Measurement: {measurement},type: {type(measurement)}"

        return measurement


class ConstantGaussianRangeNoiseModel(RangeNoiseModel):
    """This is a gaussian noise model with a constant mean and standard
    deviation. The distribution of measurements follows the following formula:

    measurement = true_distance + normal(mean, sigma)
                = normal(true_distance + mean, sigma)

    """

    def __init__(self, mean: float = 0.0, stddev: float = 0.1):
        assert isinstance(mean, float)
        assert isinstance(stddev, float)
        assert stddev > 0

        self._mean = mean
        self._stddev = stddev

        # define the measurement model
        measurement_model = lambda true_dist, timestamp: RangeMeasurement(
            true_dist,
            np.random.normal(true_dist + mean, stddev),
            mean,
            stddev,
            timestamp,
        )
        super().__init__(measurement_model)

    def __str__(self):
        return (
            f"Constant Gaussian Range Model\n"
            + f"Mean: {self._mean}\n"
            + f"Std Dev: {self._stddev}\n"
        )

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def stddev(self) -> float:
        return self._stddev


class VaryingMeanGaussianRangeNoiseModel(RangeNoiseModel):
    """This is a gaussian noise model with a mean that varies by distance and
    constant standard deviation. The distribution of measurements follows the
    following formula:

    measurement = normal(slope*true_distance, sigma)
    """

    def __init__(self, stddev: float = 0.1, slope: float = 1.05):
        assert isinstance(stddev, float)
        assert isinstance(slope, float)
        self._slope = slope
        self._stddev = stddev

        # define the measurement_model
        measurement_model = lambda true_dist, timestamp: RangeMeasurement(
            true_dist,
            np.random.normal(slope * true_dist, stddev),
            (slope - 1) * true_dist,
            stddev,
            timestamp,
        )
        super().__init__(measurement_model)

    def __str__(self):
        return (
            f"Dist-Varying Gaussian Range Model\n"
            + f"Slope: {self._slope}\n"
            + f"Std Dev: {self._stddev}\n"
        )

    @property
    def slope(self) -> float:
        return self._slope

    @property
    def stddev(self) -> float:
        return self._stddev
