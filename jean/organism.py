import torch
import torch.nn as nn
import torch.functional as F
from typing import TypedDict, List
import random
import copy
import numpy as np
from uuid import uuid4


class Arch(TypedDict):
    d_in: int
    d_out: int
    fcs: List[int]
    classifier: bool
    arch_id: str


class Organism(nn.Module):
    def __init__(self, arch: Arch, device=torch.device("cpu")):
        super(Organism, self).__init__()

        self.arch = arch
        self.arch_id = arch.get("arch_id")
        self.device = device
        self.loss: List[torch.Tensor] = []
        self.avg_loss = None
        self.build_arch(arch)
        self.to(device)

    def build_arch(self, arch):
        d_in = arch.get("d_in")
        d_out = arch.get("d_out")
        self.fcs = [*arch.get("fcs")]

        self.fcs.insert(0, d_in)
        # self.fcs.insert(-1, d_out)

        self.layers = []
        self.activations = []
        self.final_activation = (
            nn.Softmax().to(self.device) if arch.get("classifier") else None
        )

        for i, fc in enumerate(self.fcs[:-1]):
            layer = nn.Linear(fc, self.fcs[i + 1]).to(self.device)
            nn.init.uniform_(layer.weight, -2, 2)
            activ = nn.ReLU().to(self.device)
            self.layers.append(layer)
            self.activations.append(activ)

        self.out = nn.Linear(self.fcs[-1], d_out).to(self.device)

    def clone(self):
        state_dict = self.state_dict()
        child = Organism(arch=copy.deepcopy(self.arch), device=self.device)
        child.load_state_dict(state_dict)
        return child

    def forward(self, x):
        for i, fc in enumerate(self.layers):
            activation = self.activations[i]
            x = activation(fc(x))

        final_layer = self.out(x)

        if self.final_activation:
            return self.final_activation(final_layer)

        return final_layer

    def add_loss(self, loss_value: torch.Tensor):
        self.loss += loss_value

    def make_avg_loss(self):
        self.avg_loss = torch.mean(torch.Tensor(self.loss).to(self.device))
        return self.avg_loss

    def set_arch(self, arch_id: str):
        self.arch_id = arch_id

    def is_monarch(self, delegate):
        def extract_requirements(arch: Arch):
            return {
                "d_in": arch.get("d_in"),
                "d_ff": arch.get("fcs"),
                "d_out": arch.get("d_out"),
            }

        monarch = extract_requirements(self.arch) == extract_requirements(delegate.arch)

        if monarch:
            delegate.set_arch(self.arch_id)

        return monarch

    def mutate(
        self,
        mutation_rate: float,
        cross_rate: float,
        mutation_magnitude: float,
        delegate=None,
    ):
        torch.set_grad_enabled(False)

        new_org: Organism = None

        if random.random() < cross_rate and self.is_monarch(delegate=delegate):
            child = Organism(arch=self.arch, device=self.device)

            for c_param, parent1_param, parent2_param in zip(
                child.parameters(), self.parameters(), delegate.parameters()
            ):
                mask = torch.rand(c_param.size()) < 0.5
                mask = mask.to(self.device)
                c_param.data.copy_(
                    torch.where(mask, parent1_param.data, parent2_param.data)
                )
            new_org = child
        else:
            new_org = self.clone()

        for param in new_org.parameters():
            if torch.rand(1) < mutation_rate:
                noise = torch.rand(param.size()).to(self.device) * mutation_magnitude
                param.data += noise

        return new_org

    def arch_mutate(self, min_size, max_size, max_layers):
        increase_complexity_rate = 0.8
        increase_complexity = random.random() < increase_complexity_rate

        fcs = [*self.arch.get("fcs")]

        location = 0 if len(fcs) == 0 else random.randint(0, len(fcs) - 1)

        if increase_complexity:
            if len(fcs) >= max_layers:
                return self
            size = random.randint(min_size, max_size)
            fcs.insert(location, size)
        else:
            if len(fcs) < 1:
                return self
            fcs.pop(location)

        newarch = copy.deepcopy(self.arch)
        newarch["fcs"] = fcs
        newarch["arch_id"] = uuid4()
        return Organism(newarch, self.device)
