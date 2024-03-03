from granby.arch.arch import QArchFF, BagOptimizer, Bag, Unbag
from granby.envs.cube_volume import CubeVolume
import torch.nn.functional as F
import torch

def define_bag() -> Bag:
    return {
        "activation_out": lambda x: F.relu(x),
        "alpha": 0.0005,
        "drift_loss": None,
        "epsilon": 0.2,
        "features_in": 3,
        "features_out": 1,
        "gamma": 0,
        "layers": [
            { "activation": F.relu, "d": 12 },
            { "activation": F.relu, "d": 60 },
            { "activation": F.relu, "d": 120 },
            { "activation": F.relu, "d": 30 },
        ],
        "meta": {
            "populus": 30,
            "kill_epochs": 10000,
            "mutation_rate": 0.1,
            "shorten_rate": 0.01
        },
        "momentum": 0.8
    }

def make_model(bag: Bag) -> QArchFF:
    return QArchFF(bag)

if __name__ == "__main__":
    env = CubeVolume()
    bag = define_bag()
    model = make_model(bag=bag)

    print("basic training")

    for _ in range(10000):
        model.q(env.state(), env.act, pack_loss=lambda x: print(x))

    ta = model.i(torch.tensor([3,3,3], dtype=torch.float32).to(torch.device("cuda")))

    print(ta)