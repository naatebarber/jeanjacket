import numpy as np
import torch
from typing import List
import copy
import random

from jean.organism import Organism
import matplotlib.pyplot as plt

# Device setup
dev = "cpu"
if torch.cuda.is_available():
    dev = "cuda:0"
device = torch.device(dev)
global_org_cfg = {
    "d_in": 10,
    "d_out": 1,
    "d_ff": 12,
    "device": device
}
global_lr = 0.05
global_cr = 0.3

torch.set_grad_enabled(False)


def define_data():
    input_size = [10]
    input_data = torch.randn(input_size).to(device)
    return input_data

def fitness(organism: Organism, input_data: torch.Tensor):
    highest_v = torch.max(input_data).to(device)
    ans = torch.square(highest_v).to(device)

    y = organism.forward(input_data)

    loss = torch.square(ans - y)
    return loss

def mutate(organism: Organism, mutation_rate: float, cross_rate: float, delegate: Organism=None) -> Organism:
    new_org: Organism = None

    if random.random() < cross_rate:
        child = Organism(**global_org_cfg)
        for c_param, parent1_param, parent2_param in zip(child.parameters(), organism.parameters(), delegate.parameters()):
            mask = torch.rand(c_param.size()) < 0.5
            mask = mask.to(device)
            c_param.data.copy_(torch.where(mask, parent1_param.data, parent2_param.data))
        new_org = child
    else:
        new_org = copy.deepcopy(organism)

    for param in new_org.parameters():
        if torch.rand(1) < mutation_rate:
            noise = torch.rand(param.size()).to(device) * 0.1
            param.data += noise

    return new_org

def build_population(size: int, candidates: List[Organism] = None):
    if not candidates:
        return [ Organism(**global_org_cfg) for _ in range(size) ]
    else:
        next_population = []
        iterator = 0
        while len(next_population) < size:
            org_ix_a = iterator % len(candidates)
            org_ix_b = (iterator + 1) % len(candidates)
            org_a = candidates[org_ix_a]
            org_b = candidates[org_ix_b]
            next_population.append(mutate(org_a, global_lr, global_cr, org_b))
            iterator += 1
        return next_population

def main():
    population_size = 100
    epochs = 50
    n_iters_per_org = 100

    population = build_population(population_size)

    history = {
        "avg_loss_over_generations": [],
        "peak_loss_over_generations": []
    }

    for epoch in range(epochs):
        input_datas = [define_data() for _ in range(n_iters_per_org)]
        performances = {}

        for i in range(len(population)):
            org = population[i]
            performance = torch.Tensor([ fitness(org, input_datas[i]) for i in range(n_iters_per_org) ])
            avg_performance = torch.mean(performance)
            performances[i] = avg_performance

        avg_population_loss = np.mean([v for (k, v) in performances.items()])
        print(f"Average population loss: {avg_population_loss}")
        history['avg_loss_over_generations'].append(avg_population_loss)

        sorted_by_perf = sorted(performances.items(), key=lambda x: x[1])
        top_ten = sorted_by_perf[0:10]
        top_performers = [population[e[0]] for e in top_ten]
        top_performances = [e[1] for e in top_ten]

        top_5_performers = top_performers[0:5]
        
        
        peak_population_loss = np.mean(top_performances)
        print(f"Peak population loss: {peak_population_loss}")
        history['peak_loss_over_generations'].append(peak_population_loss)

        best_loss = sorted_by_perf[0][1]
        print("Best performance:", best_loss)

        population = top_5_performers + build_population(population_size - 5, top_performers)

    for i in range(len(population)):
        org = population[i]
        torch.save(org.state_dict(), f"./jean/generation/{i}-org")

    avg_loss_over_generations = history.get("avg_loss_over_generations")
    plt.figure(figsize=(10, 6))  # You can adjust the size of the figure
    plt.plot(avg_loss_over_generations, label='Average Loss')
    plt.xlabel('Generation')
    plt.ylabel('Average Loss')
    plt.title('Average Loss Over Generations')
    plt.legend()
    plt.grid(True)
    plt.show()

main()