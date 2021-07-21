from abc import abstractmethod

class NoiseModel:

    def __init__(self, name: str):
        assert isinstance(name, str)
        self._name = name

    def __str__(self):
        return self._name
