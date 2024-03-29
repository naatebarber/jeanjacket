from abc import ABC, abstractmethod
from typing import Tuple, Any
import torch


class Env(ABC):
    def __init__(self):
        self.current_state = None

    @abstractmethod
    def state(self) -> torch.TensorType:
        pass

    @abstractmethod
    def rule(self, state: torch.TensorType, action: torch.TensorType) -> Any:
        pass

    @abstractmethod
    def act(self, action: torch.TensorType) -> Tuple[Any, torch.TensorType]:
        pass
