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
        self.current_state = torch.tensor(
            data=np.random.rand(1, 3), dtype=torch.float32
        )
        return self.current_state

    def rule(self, state: TensorType, action: TensorType) -> Any:
        result: TensorType = torch.prod(state, 1).squeeze()
        action = action.squeeze()

        loss = torch.square(result - action)
        print(result, action)
        print(loss)

        return loss

    def act(self, action: TensorType) -> Tuple[Any, TensorType]:
        loss = self.rule(self.current_state, action)
        return (loss, self.current_state())
