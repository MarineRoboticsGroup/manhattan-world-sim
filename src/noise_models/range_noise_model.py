import numpy as np
from typing import Callable

from src.noise_models.noise_model import NoiseModel


class RangeNoiseModel(NoiseModel):
    """
    Base range noise model. Takes in a function as an argument which describes
    the sensor model as follows:

    measurement = f(true_distance, sensor_params)
    """

    def __init__(self, name: str, noise_func: Callable[float]):
        assert isinstance(name, str)
        assert isinstance(noise_func, Callable)
        super().__init__(name)
        self._noise_func = noise_func

    def __str__(self):
        return (
            f"Generic Range Model: {self._name}\n"
            + f"Mean: {self._mean}\n"
            + f"Std Dev: {self._stddev}\n"
        )

    @property
    def measurement_model(self):
        return self._measurement_model

    def get_range_measurement(self, true_dist: float) -> float:
        measurement = self._measurement_model(true_dist)
        assert isinstance(measurement, float)
        if measurement < 0:
            return 0
        else:
            return measurement


class ConstantGaussianRangeNoiseModel(RangeNoiseModel):
    """This is a gaussian noise model with a constant mean and standard
    deviation. The distribution of measurements follows the following formula:

    measurement = true_distance + normal(mean, sigma)
                = normal(true_distance + mean, sigma)

    """

    def __init__(self, name: str, mean: float, stddev: float):
        assert isinstance(name, str)
        assert isinstance(mean, float)
        assert isinstance(stddev, float)
        assert stddev > 0

        self._mean = mean
        self._stddev = stddev

        # define the measurement model
        measurement_model = lambda true_dist: np.random.normal(true_dist + mean, stddev)
        super().__init__(name, measurement_model)

    def __str__(self):
        return (
            f"Constant Gaussian Range Model: {self._name}\n"
            + f"Mean: {self._mean}\n"
            + f"Std Dev: {self._stddev}\n"
        )

    @property
    def mean(self):
        return self._mean

    @property
    def stddev(self):
        return self._stddev


class VaryingMeanGaussianRangeNoiseModel(RangeNoiseModel):
    """This is a gaussian noise model with a mean that varies by distance and
    constant standard deviation. The distribution of measurements follows the
    following formula:

    measurement = normal(slope*true_distance, sigma)
    """

    def __init__(self, name: str, stddev: float, slope: float):
        assert isinstance(name, str)
        assert isinstance(stddev, float)
        assert isinstance(slope, float)
        self._slope = slope
        self._stddev = stddev

        # define the measurement_model
        measurement_model = lambda true_dist: np.random.normal(slope*true_dist, stddev)
        super().__init__(name, measurement_model)

    def __str__(self):
        return (
            f"Dist-Varying Gaussian Range Model: {self._name}\n"
            + f"Slope: {self._slope}\n"
            + f"Std Dev: {self._stddev}\n"
        )

    @property
    def slope(self):
        return self._slope

    @property
    def stddev(self):
        return self._stddev
