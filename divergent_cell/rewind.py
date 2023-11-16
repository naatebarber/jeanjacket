from typing import List
from divergent_cell.transport import Transport

"""
Rewind the correct transports to have output dimensions summing up to fit the output size.
"""


class Rewind:
    def __init__(self, transports: List[Transport], output_size: int):
        self.transports = transports
        self.output_size = output_size

    def rewind(self):
        pass
