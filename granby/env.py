from abc import ABC, abstractmethod
from typing import Tuple, Any
import torch


class Env(ABC):
    @abstractmethod
    def state() -> torch.TensorType:
        pass

    @abstractmethod
    def act(action: torch.TensorType) -> Tuple[Any, torch.TensorType]:
        pass
