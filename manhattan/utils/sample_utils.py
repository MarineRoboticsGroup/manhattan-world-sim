import numpy as np
from typing import Iterable

def choice(a: Iterable[object]) -> object:
    choices = [x for x in range(len(a))]
    choice_idx = np.random.choice(choices)
    return a[choice_idx]