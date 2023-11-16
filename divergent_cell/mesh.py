from typing import TypedDict, List, Tuple
from divergent_cell.cell import Cell, CellConfig, Merge, Split
from divergent_cell.transport import Transport
import random
import numpy as np


class MeshConfig(TypedDict):
    cell_count: int
    max_neighbors: int
    split_tendency: float
    merge_tendency: float
    cell_config: CellConfig


class Mesh:
    def __init__(self, config: MeshConfig, inputs: int, outputs: int):
        self.cell_count = config.get("cell_count")
        self.max_neighbors = config.get("max_neighbors")
        self.max_hops = config.get("max_hops")
        self.split_tendency = config.get("split_tendency")
        self.merge_tendency = config.get("merge_tendency")
        self.cell_config = config.get("cell_config")

        self.inputs = inputs
        self.outputs = outputs

        self.cells: List[Cell] = []
        self.transports: List[Transport] = []

    def make_mesh(self):
        self.cells = []
        self.weights = np.random.rand(self.cell_count) * 4 - 2
        self.biases = np.random.rand(self.cell_count) * 4 - 2

        for i in range(self.cell_count):
            split = random.random() < self.split_tendency
            merge = random.random() < self.merge_tendency

            if split:
                self.cells.append(
                    Split(self.cell_config, self.weights[i], self.biases[i])
                )
            elif merge:
                self.cells.append(
                    Merge(self.cell_config, self.weights[i], self.biases[i])
                )
            else:
                self.cells.append(
                    Cell(self.cell_config, self.weights[i], self.biases[i])
                )

        for i in range(self.cell_count - 1):
            a = self.cells[i]
            b = self.cells[i + 1]
            a.next_cell = b

        first_in_chain = self.cells[0]

        for cell in self.cells:
            first_in_chain.befriend(cell)

        return first_in_chain

    def feed_mesh(self, inputs: List[float], output_length: int):
        self.transports = [Transport([x]) for x in inputs]

        def current_size(ts: List[Transport]):
            return sum([len(t.values) for t in ts])

        state: List[Tuple[Transport, Cell]] = []
        for t in self.transports:
            state.append([t, self.cells[0]])

        while current_size(self.transports) != output_length:
            if len(state) is 0:
                print("Failed to find a pathway")
                break

            for i, (t, c) in enumerate(state):
                if c:
                    state[i] = c.apply(t)
                else:
                    del state[i]

        return self.transports
