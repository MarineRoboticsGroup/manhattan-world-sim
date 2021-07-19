from .noise_model import NoiseModel

class RangeNoiseModel(NoiseModel):
    """
    Range noise model.
    """

    def __init__(self, mean: float, stddev: float):
        """
        Initialize the range noise model.
        :param mean: The mean.
        :param stddev: The standard deviation.
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def get_noise(self, measurement):
        """
        Get the noise for the given measurement.
        :param measurement: The measurement.
        :return: The noise.
        """
        return measurement.get_range_noise()