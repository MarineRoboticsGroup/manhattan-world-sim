import numpy as np
from typing import Sequence, Sized, overload, Tuple, List
from manhattan.geometry.TwoDimension import Point2, SE2Pose
from manhattan.agent.agent import Robot, Beacon


@overload
def choice(a: List[Tuple[Point2, float]]) -> Tuple[Point2, float]:
    pass


@overload
def choice(a: List[SE2Pose]) -> SE2Pose:
    pass

@overload
def choice(a: List[Point2]) -> Point2:
    pass

@overload
def choice(a: List[Beacon]) -> Beacon:
    pass

@overload
def choice(a: List[str]) -> str:
    pass


def choice(a: Sequence[object]) -> object:
    choices = [x for x in range(len(a))]
    choice_idx = np.random.choice(choices)
    sample = a[choice_idx]
    return sample
