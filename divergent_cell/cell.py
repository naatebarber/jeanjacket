from typing import TypedDict, List
import numpy as np
import random
import divergent_cell.transport as t


class CellConfig(TypedDict):
    charisma: float
    max_hops: int


class Cell:
    def __init__(self, config: CellConfig, w: float, b: float):
        self.charisma = config.get("charisma")
        self.max_hops = config.get("max_hops")

        self.narcisist = False

        self.weight = w
        self.bias = b
        self.activation = lambda x: x if x > 0 else x * 0.1
        self.clique: List[Cell] = []
        self.next_cell: Cell = None

    def befriend(self, cell):
        if random.random() < self.charisma and self is not cell:
            self.clique.append(cell)

        if self.next_cell:
            self.next_cell.befriend(cell)

    def apply(self, t: t.Transport):
        if t.hops > self.max_hops:
            return t, None

        apply_mutation = lambda v: self.activation((self.weight * v) + self.bias)
        new_values = [apply_mutation(v) for v in t.values]
        t.update(self, new_values=new_values)

        if len(self.clique) > 0:
            next_cell: Cell = random.choice(self.clique)
            return t, next_cell
        else:
            return t, None


class Split(Cell):
    def apply(self, t: t.Transport):
        if t.hops > self.max_hops:
            return t, None

        apply_mutation = lambda v: self.activation((self.weight * v) + self.bias)
        new_values = [apply_mutation(v) for v in t.values]
        x = new_values[-1]
        split = apply_mutation(x)
        new_values.append(split)

        t.update(self, new_values=new_values)

        if len(self.clique) > 0:
            next_cell: Cell = random.choice(self.clique)
            return t, next_cell
        else:
            return t, None


class Merge(Cell):
    def apply(self, t: t.Transport):
        if t.hops > self.max_hops:
            return t, None

        apply_mutation = lambda v: self.activation((self.weight * v) + self.bias)
        new_values = [apply_mutation(v) for v in t.values]

        if len(new_values) > 1:
            new_values[0] = new_values[0] + new_values[1]
            new_values.pop(1)

        t.update(self, new_values=new_values)

        if len(self.clique) > 0:
            next_cell: Cell = random.choice(self.clique)
            return t, next_cell
        else:
            return t, None
