import asyncio
import copy
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from jean.organism import Organism

# Device setup
dev = "cpu"
if torch.cuda.is_available():
    dev = "cuda:0"
device = torch.device(dev)
global_org_cfg = {"d_in": 10, "d_out": 1, "d_ff": 12, "device": device}
global_lr = 0.05
global_cr = 0.3
global_mm = 0.1

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


def mutate(
    organism: Organism,
    mutation_rate: float,
    cross_rate: float,
    delegate: Organism = None,
) -> Organism:
    new_org: Organism = None

    if random.random() < cross_rate:
        child = Organism(**global_org_cfg)

        for c_param, parent1_param, parent2_param in zip(
            child.parameters(), organism.parameters(), delegate.parameters()
        ):
            mask = torch.rand(c_param.size()) < 0.5
            mask = mask.to(device)
            c_param.data.copy_(
                torch.where(mask, parent1_param.data, parent2_param.data)
            )
        new_org = child
    else:
        new_org = copy.deepcopy(organism)

    for param in new_org.parameters():
        if torch.rand(1) < mutation_rate:
            noise = torch.rand(param.size()).to(device) * global_mm
            param.data += noise

    return new_org


def build_population(size: int, candidates: List[Organism] = None):
    if not candidates:
        return [Organism(**global_org_cfg) for _ in range(size)]
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


def split(iterable, n: int):
    batches = [[] for _ in range(n)]
    for i, x in enumerate(iterable):
        batches[i % n].append(x)

    return batches


def join(i: List[List[Any]]):
    tot = []
    for x in i:
        tot += x
    return tot


async def main():
    global global_mm
    global global_lr
    global global_cr

    population_size = 50
    epochs = 200
    n_iters_per_org = 100
    # Carry over n% of the previous population
    population_carryover = 0.05
    procs = os.cpu_count()

    population = build_population(population_size)

    history = {
        "avg_loss_over_generations": [],
        "peak_loss_over_generations": [],
        "best_loss_over_generations": [],
    }
    pool = ThreadPoolExecutor(max_workers=procs)

    for epoch in range(epochs):
        performances = {}

        input_datas = [define_data() for _ in range(n_iters_per_org)]
        input_datas = split(input_datas, procs)
        subpopulations = split(population, procs)

        def evaluate_subpopulation(sp: List[Organism], id: List[torch.Tensor]):
            subpopulation_performance = {}
            for i in range(len(sp)):
                org = sp[i]
                performance = torch.Tensor([fitness(org, i) for i in id])
                avg_performance = torch.mean(performance)
                subpopulation_performance[i] = avg_performance
            return (len(sp), subpopulation_performance)

        parallels = await asyncio.gather(
            *[
                asyncio.wrap_future(
                    pool.submit(
                        evaluate_subpopulation, subpopulations[i], input_datas[i]
                    )
                )
                for i in range(len(subpopulations))
            ]
        )

        offset = 0
        for parallel in parallels:
            length, performance = parallel
            performance = {k + offset: v for k, v in performance.items()}
            performances = {**performances, **performance}
            offset += length

        avg_population_loss = np.mean([v for (k, v) in performances.items()])
        print(f"Average population loss: {avg_population_loss}")
        history["avg_loss_over_generations"].append(avg_population_loss)

        sorted_by_perf = sorted(performances.items(), key=lambda x: x[1])
        num_to_carry_over = int(len(sorted_by_perf) * population_carryover)
        top = sorted_by_perf[0:num_to_carry_over]

        top_performers = [population[e[0]] for e in top]
        top_performances = [e[1] for e in top]

        peak_population_loss = np.mean(top_performances)
        print(f"Peak population loss: {peak_population_loss}")
        history["peak_loss_over_generations"].append(peak_population_loss)

        best_loss = sorted_by_perf[0][1]
        print("Best performance:", best_loss)
        history["best_loss_over_generations"].append(best_loss)

        population = top_performers + build_population(
            population_size - len(top_performers), top_performers
        )

        if global_mm > 0:
            global_mm *= 0.99

        if global_cr < 1:
            global_cr /= 0.99

    for i in range(len(population)):
        org = population[i]
        torch.save(org.state_dict(), f"./jean/generation/{i}-org")

    avg_loss_over_generations = history.get("avg_loss_over_generations")
    best_loss_over_generations = history.get("best_loss_over_generations")
    plt.figure(figsize=(10, 6))  # You can adjust the size of the figure
    plt.plot(avg_loss_over_generations, label="Average Loss")
    plt.plot(best_loss_over_generations, label="Best Loss")
    plt.xlabel("Generation")
    plt.ylabel("Average Loss")
    plt.title("Average Loss Over Generations")
    plt.legend()
    plt.grid(True)
    plt.show()


asyncio.run(main())
