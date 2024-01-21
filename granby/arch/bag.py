from typing import TypedDict, Callable, Any, List
import torch

class Layer(TypedDict):
    d: int
    activation: Callable[[Any], Any]

class Bag(TypedDict):
    features_in: int
    layers: List[Layer]
    activation_out: Callable[[Any], Any]
    features_out: int
    epsilon: float
    gamma: float
    alpha: float
    cascade: torch.TensorType