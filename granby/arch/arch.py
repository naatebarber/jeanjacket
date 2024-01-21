from typing import TypedDict, Callable, Any, List, Tuple
from copy import deepcopy
from abc import ABC, abstractmethod
import random
import torch
from torch._C import TensorType
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from granby.env import Env


class Meta(TypedDict):
    populus: int
    kill_epochs: int
    mutation_rate: float
    shorten_rate: float


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
    drift_loss: torch.TensorType
    meta: Meta


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


class QArch(ABC):
    @abstractmethod
    def __init__(self, bag: Bag):
        pass

    @abstractmethod
    def q(
        self,
        state: torch.TensorType,
        act: Callable[[torch.TensorType], Tuple[Any, torch.TensorType]],
    ):
        pass

    @abstractmethod
    def m(self, state: torch.TensorType, desired: torch.TensorType):
        pass

    @abstractmethod
    def i(self, state: torch.TensorType) -> torch.TensorType:
        pass


class QArchFF(nn.Module, QArch):
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

    def q(
        self,
        state: torch.TensorType,
        act: Callable[[torch.TensorType], Tuple[Any, torch.TensorType]],
    ):
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

        return loss.detach()

    def m(self, state: TensorType, desired: TensorType):
        current_q_values = self.forward(state)
        current_q_max = current_q_values.max(1)[0]

        desired_q_max = desired.max(1)[0]
        loss = F.mse_loss(current_q_max, desired_q_max)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.detach()

    def i(self, state: torch.TensorType) -> TensorType:
        return self.forward(state)


class Diluted(TypedDict):
    model: QArch
    bag: Bag
    losses: List[torch.TensorType]
    done: bool


class BagOptimizer:
    def __init__(self, bag: Bag, env: Env, model: QArch):
        self.bag = bag
        self.env = env
        self.model = model

        meta = bag.get("meta")
        self.drift_loss = bag.get("drift_loss")
        self.populus = meta.get("populus")
        self.kill_epochs = meta.get("kill_epochs")
        self.mutation_rate = meta.get("mutation_rate")
        self.shorten_rate = meta.get("shorten_rate")

        self.generation: List[Diluted] = []

    def initial_train(self):
        last_loss = None
        while last_loss is not None and last_loss > self.drift_loss:
            state = self.env.state()
            last_loss = self.model.q(state, self.env.act)

    def optimize_offload(
        self, gen_q: Callable[[Bag], QArch], gen_state: Callable[[], torch.TensorType]
    ):
        gen_pop = []
        for _ in range(self.populus):
            decision = random.random()
            if decision < self.shorten_rate:
                gen_pop.append(BagOptimizer.shorten(self.bag))
                continue
            gen_pop.append(BagOptimizer.dilute(self.bag, self.mutation_rate))

        self.generation = [
            {"bag": bag, "losses": [], "model": gen_q(bag), "done": False}
            for bag in gen_pop
        ]

        for _ in range(self.kill_epochs):
            state = gen_state()
            prelude = self.model.i(state)

            for diluted in [g for g in self.generation if g.get("done") is False]:
                m = diluted.get("model")
                m_loss = m.m(state, desired=prelude)
                diluted["losses"].append(m_loss)

                losses = diluted.get("losses")

                qualifies = torch.mean(losses[-10:]) < self.drift_loss
                if qualifies:
                    diluted["done"] = True

        successful_diluted = [g for g in self.generation if g.get("done") is False]

        if len(successful_diluted) == 0:
            return self.model

        diluted_min_loss = successful_diluted[0]
        for diluted in successful_diluted:
            if diluted.get("loss") < diluted_min_loss.get("loss"):
                diluted_min_loss = diluted

        self.model = diluted_min_loss.get("model")
        self.bag = diluted_min_loss.get("bag")
        return self.optimize_offload(gen_q, gen_state)

    @staticmethod
    def dilute(bag: Bag, mutation_rate: float):
        # Shrink the number of neurons between hidden layer m, and hidden layer n by mutation_rate
        dilute_bag = deepcopy(bag)
        layers = len(dilute_bag.get("layers"))
        dilute_layer_ix = random.randint(0, layers)
        initial_neurons = dilute_bag["layers"][dilute_layer_ix]["d"]
        dilute_bag["layers"][dilute_layer_ix]["d"] -= int(
            initial_neurons * (random.random() * mutation_rate)
        )
        return dilute_bag

    @staticmethod
    def shorten(bag: Bag, mutation_rate: float):
        # Remove hidden layer n from bag, far more drastic
        shortcake_bag = deepcopy(bag)
        layers = len(shortcake_bag.get("layers"))

        if layers < 2:
            return shortcake_bag

        pop_layer_ix = random.randint(0, layers)
        shortcake_bag["layers"].pop(pop_layer_ix)

        return shortcake_bag
