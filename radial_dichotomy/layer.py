import math
from radial_dichotomy.sym import Symbol


class Layer:
    def __init__(self, symbols=[]):
        self.symbols = symbols

    def fuse(parent_a, parent_b):
        symbols = [
            [Symbol(sym_a, sym_b) for sym_b in parent_b.symbols]
            for sym_a in parent_a.symbols
        ]
        symbols = [sym for symbol in symbols for sym in symbol]

        return Layer(symbols=symbols)

    def merge(self, other):
        self.symbols = [*self.symbols, *other.symbols]
        del other
        return self
