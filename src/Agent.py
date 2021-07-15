from typing import Hashable
from enum import Enum
from typing import List, Set
import numpy as np

from geometry.TwoDimension import SE2Pose


class AgentType(Enum):
    Robot = "Robot"
    Beacon = "Beacon"

class Agent(object):
    def __init__(self, name: Hashable, type: AgentType = AgentType.Robot):
        self._name = name
        self._type = type

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type
    def __str__(self) -> str:
        return " ".join([self._type,
                         str(self.name)])

    def __hash__(self) -> int:
        return hash(self._type+str(self._name))

class Robot(Agent):
    def __init__(self, name, range_std: float, odom_cov: np.ndarray):
        super().__init__(name)
        self._range_std = range_std
        self._odom_cov = odom_cov

    def get_range_measurement(self, gt_range: float):
        return np.random.normal(loc=gt_range, scale=self._range_std)

    def get_odom_measurement(self, gt_rel_pose: SE2Pose):
        lie_noise = np.random.multivariate_normal([0,0,0],self._odom_cov)
        return gt_rel_pose * SE2Pose.by_exp_map(lie_noise)

class Beacon(Agent):
    def __init__(self, name):
        super().__init__(name, AgentType.Beacon)