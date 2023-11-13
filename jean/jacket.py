import copy
import random
import torch
import os
import asyncio
from typing import TypedDict, List, Callable, Dict
from uuid import uuid4, UUID
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from jean.organism import Organism, Arch


class JacketConfig(TypedDict):
    classifier: bool
    inputs: int
    outputs: int
    max_layers: int
    max_layer_size: int
    min_layer_size: int
    carryover_rate: float
    mutation_rate: float
    mutation_magnitude: float
    crossover_rate: float
    arch_mutation_rate: float
    decay: float
    population_size: int


class Jacket:
    def __init__(self, **config: JacketConfig):
        dev = "cpu"
        if torch.cuda.is_available():
            dev = "cuda"
        self.device = torch.device(dev)

        self.thread_pool = ThreadPoolExecutor(os.cpu_count())
        self.process_pool = ProcessPoolExecutor(os.cpu_count())

        self.classifier = config.get("classifier")
        self.max_layers = config.get("max_layers")
        self.max_size = config.get("max_layer_size")
        self.min_size = config.get("min_layer_size")
        self.carryover_rate = config.get("carryover_rate")
        self.crossover_rate = config.get("crossover_rate")
        self.mutation_rate = config.get("mutation_rate")
        self.mutation_magnitude = config.get("mutation_magnitude")
        self.arch_mutation_rate = config.get("arch_mutation_rate")
        self.decay = config.get("decay")
        self.population_size = config.get("population_size")

        self.source_arch: Arch = {
            "classifier": self.classifier,
            "d_in": config.get("inputs"),
            "d_out": config.get("outputs"),
            "fcs": [],
            "arch_id": uuid4(),
        }

        self.population: List[Organism] = []
        self.loss_fn: Callable[[Organism, torch.Tensor]] = None

    def loss(self, loss_fn: Callable[[Organism, torch.Tensor], torch.Tensor]):
        self.loss_fn = loss_fn

    def sort_by_fitness(self, population: List[Organism]):
        population.sort(key=lambda org: org.avg_loss)
        return population

    def group_by_arch(self, population: List[Organism]):
        archetypes: Dict[UUID, List[Organism]] = {}
        for org in population:
            if org.arch_id in archetypes:
                archetypes[org.arch_id].append(org)
            else:
                archetypes[org.arch_id] = [org]

        return archetypes

    def evolve(self):
        source_organisms = None
        if len(self.population) > 0:
            n_carryover = int(len(self.population) * self.carryover_rate)
            source_organisms = self.population[:n_carryover]

        if not source_organisms:
            self.population = [
                Organism(self.source_arch, device=self.device)
                for _ in range(self.population_size)
            ]
            return self.population
        else:
            new_population: List[Organism] = [*source_organisms]

            archetypes = self.group_by_arch(source_organisms)
            # print(archetypes.keys())

            weight_potential: Dict[UUID, torch.Tensor] = {}
            for arch_id, orgs in archetypes.items():
                self.sort_by_fitness(orgs)
                arch_avg_fitness = torch.mean(
                    torch.stack([org.avg_loss for org in orgs])
                )
                weight_potential[arch_id] = arch_avg_fitness

            sum_loss = torch.sum(
                torch.stack([loss for _, loss in weight_potential.items()])
            )
            avg_loss = torch.mean(
                torch.stack([loss for _, loss in weight_potential.items()])
            )

            sum_contrib = None
            for arch_id, loss in weight_potential.items():
                # Architectures with a lower loss ratio are weighted as having higher potential
                x = loss / sum_loss
                arch_contrib = 1 / (x + 1)

                weight_potential[arch_id] = arch_contrib
                if not sum_contrib:
                    sum_contrib = arch_contrib
                else:
                    sum_contrib = torch.sum(torch.stack([sum_contrib, arch_contrib]))

            new_orgs_per_arch = {
                arch_id: int(
                    (arch_contrib / sum_contrib)
                    * (self.population_size - len(source_organisms))
                )
                for arch_id, arch_contrib in weight_potential.items()
            }

            for arch_id, orgs in archetypes.items():
                n = new_orgs_per_arch[arch_id]
                for i in range(n):
                    parent_ix = i % len(orgs)
                    parent = orgs[parent_ix]
                    delegate_ix = (i + 1) % len(orgs)
                    delegate = orgs[delegate_ix]

                    mutate_arch = random.random() < self.arch_mutation_rate
                    child = None
                    if mutate_arch:
                        child = parent.arch_mutate(
                            self.min_size, self.max_size, self.max_layers
                        )
                    else:
                        child = parent.mutate(
                            self.mutation_rate,
                            self.crossover_rate,
                            self.mutation_magnitude,
                            delegate,
                        )

                    new_population.append(child)

            self.population = new_population
            self.arch_mutation_rate *= self.decay
            self.mutation_rate *= self.decay

    def split_population(self, n_subpopulations: int):
        subpopulations = [[] for _ in range(n_subpopulations)]
        for i, org in enumerate(self.population):
            subpopulations[i % n_subpopulations].append(org)

        return subpopulations

    async def eval(self, input_batch: List[torch.Tensor]):
        if not self.loss_fn:
            raise Exception(
                "You need to provide a loss function using the Jacket.loss() method."
            )

        subpopulations = self.split_population(os.cpu_count())

        def evaluate_subpopulation(
            subpopulation: List[Organism],
            input_batch: List[torch.Tensor],
            loss_fn: Callable[[Organism, torch.Tensor], torch.Tensor],
        ):
            for org in subpopulation:
                for data in input_batch:
                    loss = loss_fn(org, data)
                    loss = loss.to(self.device)
                    org.add_loss(loss)

        # [ evaluate_subpopulation(subpopulation, input_batch, self.loss_fn) for subpopulation in subpopulations ]
        evaluate_subpopulation(self.population, input_batch, self.loss_fn)

        # await asyncio.gather(*[
        #     asyncio.wrap_future(
        #         self.process_pool.submit(evaluate_subpopulation, subpopulation, copy.deepcopy(input_batch), copy.deepcopy(self.loss_fn))
        #     )
        #     for subpopulation in subpopulations
        # ])

        # self.process_pool.map(evaluate_subpopulation, [(subpopulation, input_batch) for subpopulation in subpopulations])

        self.population.sort(key=lambda x: x.make_avg_loss())
        net_loss = torch.mean(torch.stack([org.avg_loss for org in self.population]))
        best_loss = self.population[0].avg_loss
        return net_loss, best_loss
