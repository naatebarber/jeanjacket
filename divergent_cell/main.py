from divergent_cell.mesh import Mesh, MeshConfig
from divergent_cell.cell import Split
from divergent_cell.transport import Transport
import sys


def main():
    sys.setrecursionlimit(100000)
    mesh_cfg: MeshConfig = {
        "cell_count": 1000,
        "merge_tendency": 0.3,
        "split_tendency": 0.3,
        "max_neighbors": 300,
        "cell_config": {"charisma": 0.1, "max_hops": 3000},
    }

    m = Mesh(mesh_cfg, 10, 1)
    first = m.make_mesh()
    out = m.feed_mesh([1.0, 2.0, 3.0], 30)

    # Values are ending up the same because RELU will set the vector to 0, then the next bias will apply everywhere.
    # Could take care of this in two ways:
    # Make the weight + bias of cells not a scalar
    # Instead of adding dimensionality to transports, split should make a new transport and merge should combine two.

    for t in out:
        print(t.values)


if __name__ == "__main__":
    main()
