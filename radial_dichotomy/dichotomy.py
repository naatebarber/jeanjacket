from radial_dichotomy.sym import Symbol
from radial_dichotomy.layer import Layer
from typing import List
from copy import deepcopy


class RadialDichotomy:
    def __init__(self, a_priori, reverse_coupling: int, depth: int):
        self.a_priori = [Symbol(None, None).intrinsic(x) for x in a_priori]
        self.reverse_coupling = reverse_coupling
        self.depth = depth
        self.concentric: List[Layer] = []

    def permute(self, accumulate: bool = False):
        intrinsic_layer = Layer(symbols=self.a_priori)
        self.concentric.append(intrinsic_layer)
        first_pass = True

        for layer in range(self.depth):
            layer_a = self.concentric[layer]

            for reverse_couple in range(self.reverse_coupling):
                if layer - reverse_couple < 0:
                    continue

                layer_b = self.concentric[layer - reverse_couple]
                layer_c = Layer.fuse(layer_a, layer_b)

                if accumulate:
                    layer_a.merge(layer_c)
                else:
                    self.concentric.append(layer_c)

            if accumulate:
                self.concentric.append(deepcopy(layer_a))


# Prioritize convenience in training. There is no wittling down of
# neurons until all that remain are those that pertain to the problem
# only the selection of knowledge available that best fits the
# intricacies of the environment

# What that meant was when training, search for layer/symbol combinations that map closely
# to the desired output and mutate them somehow. cause remembrance
