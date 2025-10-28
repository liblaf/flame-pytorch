import sys
from pathlib import Path

import pyvista as pv
import torch
from jaxtyping import Float
from torch import Tensor

from liblaf import grapes, melon
from liblaf.flame_pytorch import FLAME


def main() -> None:
    grapes.logging.init()
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    output_dir: Path = Path(sys.argv[0]).with_suffix("")
    flame = FLAME()
    for i in range(flame.config.shape_params):
        shape: Float[Tensor, "batch shape"] = torch.zeros(
            (flame.batch_size, flame.config.shape_params)
        )
        shape[0, i] = 1.0
        vertices: Float[Tensor, "batch vertices 3"]
        vertices, _ = flame(shape=shape)
        mesh: pv.PolyData = pv.PolyData.from_regular_faces(
            vertices[0].numpy(force=True), flame.faces
        )
        melon.save(output_dir / f"shape_{i:03d}.ply", mesh)


if __name__ == "__main__":
    main()
