from typing import TypedDict, Callable, Any, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from granby.arch.bag import Bag

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
    momentum: float
    cascade: torch.TensorType

class Unbag:
    def __init__(self, bag: Bag):
        self.bag = bag
    
    def linear_trunk(self):
        d_in = self.bag.get("features_in")
        d_out = self.bag.get("features_out")
        hidden = self.bag.get("layers")
        
        p_d = d_in
        model = {}
        
        for lay in hidden:
            d = lay.get("d")
            fc = nn.Linear(p_d, d)
            fact = lambda fc: lambda x: lay.get("activation")(fc(x))
            p_d = d
            yield (fc, fact)
            
        final_fc = nn.Linear(p_d, d_out)
        final_fact = lambda fc: lambda x: self.bag.get("activation_out")(fc(x))
            
        yield (final_fc, final_fact)
        
    def hyperparam_sgd(self):
        return (self.bag.get("alpha"), self.bag.get("momentum"))
    

class QArchFF(nn.Module):
    def __init__(self, bag: Bag):
        super(QArchFF, self).__init__()
        
        contents = Unbag(bag)
        layers = contents.linear_trunk()
        
        self.forward = None
        self.gamma = bag.get("gamma")
        
        for i, lay in reversed(enumerate(layers)):
            fc = lay[0]
            fact = lay[1]
            
            setattr(self, f"fc{i}", fc)
            if not self.forward:
                self.forward = fact(fc)
            else:
                self.forward = lambda x: fact(fc(self.forward(x)))
                
        alpha, momentum = contents.hyperparam_sgd()
        self.optim = optim.SGD(self.parameters(), lr=alpha, momentum=momentum)
        
    def q(self, state: torch.TensorType, act: Callable[[torch.TensorType], Tuple[Any, torch.TensorType]]):
        current_q_values = self.forward(state)
        current_q_max = current_q_values.max(1)[0]
        
        reward, next_state = act(current_q_values)
        
        reward = F.relu(torch.Tensor([reward], dtype=torch.float32))
        
        next_q_values = self.forward(next_state)
        next_q_max = next_q_values.max(1)[0]
        
        target_q = reward + (self.gamma * next_q_max)
        loss = F.mse_loss(current_q_max, target_q)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        