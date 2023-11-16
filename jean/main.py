import asyncio
import torch
import matplotlib.pyplot as plt
from jean.jacket import Jacket, JacketConfig
from jean.organism import Organism


async def main():
    torch.set_grad_enabled(False)

    config: JacketConfig = {
        "arch_mutation_rate": 0.1,
        "carryover_rate": 0.1,
        "crossover_rate": 0.4,
        "mutation_magnitude": 0.1,
        "mutation_rate": 0.5,
        "max_layer_size": 24,
        "min_layer_size": 8,
        "max_layers": 4,
        "inputs": 10,
        "outputs": 1,
        "population_size": 100,
        "classifier": False,
        "decay": 0.995,
    }

    jacket = Jacket(**config)
    jacket.evolve()

    def loss(org: Organism, input: torch.Tensor):
        max_value = torch.max(input)
        y = torch.square(max_value)

        org_y = org.forward(input)
        loss = torch.square(y - org_y)
        return loss

    jacket.loss(loss)

    epochs = 1000

    avg_losses = []
    best_losses = []

    for _ in range(epochs):
        data = [torch.rand(10).to(jacket.device) * 10 for _ in range(10)]
        avg_loss, best_loss = await jacket.eval(data)
        avg_losses.append(avg_loss.cpu())
        best_losses.append(best_loss.cpu())
        print(avg_loss, best_loss)
        jacket.evolve()

    plt.figure(figsize=(10, 6))  # You can adjust the size of the figure
    plt.plot(avg_losses, label="Average Loss")
    plt.plot(best_losses, label="Best Loss")
    plt.xlabel("Generation")
    plt.ylabel("Average Loss")
    plt.title("Average Loss Over Generations")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    asyncio.run(main())
