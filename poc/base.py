from dataclasses import dataclass

import numpy as np


@dataclass
class RsuConfig:
    """
    Configuration of an RSU.
    """

    pos: tuple[int, int]
    range: int
    capacity: int


def distance(pos1, pos2):
    """Calculate the Euclidean distance between two positions."""

    x1, y1 = pos1
    x2, y2 = pos2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
