import math


class Symbol:
    def __init__(self, parent_a, parent_b):
        self.parent_a = parent_a
        self.parent_b = parent_b
        if parent_a and parent_b:
            self.x = parent_a.x + parent_b.x

    def intrinsic(self, x):
        self.x = x
        return self

    def place(self, radius):
        y = radius * math.sin(self.x)
        x = radius * math.cos(self.x)

        return x, y
