import random
from typing import TypedDict, List


class Transform(TypedDict):
    after: List[float]
    cell: object


class Transport:
    def __init__(self, value: List[float]):
        self.values: List[float] = value
        self.transforms: List[Transform] = []
        self.size_histogram: List[int] = []
        self.hops = 0

        self.transforms.append({"cell": None, "after": value})

    def update(self, cell, new_values: float):
        self.hops += 1
        self.transforms.append({"cell": cell, "after": new_values})
        self.size_histogram.append(len(new_values))

        self.values = new_values
        return self
