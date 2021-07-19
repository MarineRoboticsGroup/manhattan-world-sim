from abc import abstractmethod

class NoiseModel:

    def __init__(self):
        super().__init__()

    def __str__(self):
        return super().__str__()

    @abstractmethod
    def sample(self, x):
        pass

    @abstractmethod
    def log_pdf(self, x):
        pass

    @abstractmethod
    def get_covariance(self):
        pass

    @abstractmethod
    def get_inv_covariance(self):
        pass

