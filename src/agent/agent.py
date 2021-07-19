from typing import Tuple
from abc import abstractmethod

class Agent():
    """
    This class represents a general agent. In our simulator this is either a
    robot or a stationary beacon.
    """

    def __init__(self, name: str, start_loc: Tuple[int], range_model: str):
        assert isinstance(name, str)
        assert isinstance(start_loc, tuple)
        assert len(start_loc) == 2
        assert isinstance(range_model, str)

        self.name = name
        self.groundtruth_pose = start_loc


