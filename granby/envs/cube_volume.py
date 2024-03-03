from typing import Any, Tuple
from torch import TensorType
from torch._C import TensorType
import numpy as np
import torch
from granby.env import Env

class CubeVolume(Env):
    def __init__(self):
        super().__init__()

    def state(self) -> TensorType:
        self.current_state = torch.tensor(data=np.random.rand(1, 3), dtype=torch.float32).to(self.device)
        self.current_state *= 100
        return self.current_state
    
    def rule(self, state: TensorType, action: torch.Tensor) -> Any:
        result: TensorType = torch.prod(state, 1).squeeze()
        action = action.squeeze().detach()

        loss = torch.square(result - action)
        return 1 / loss

    def act(self, action: TensorType) -> Tuple[Any, TensorType]:
        loss = self.rule(self.current_state, action)
        return (loss, self.state())