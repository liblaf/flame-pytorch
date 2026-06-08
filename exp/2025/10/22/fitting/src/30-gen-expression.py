from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from jaxtyping import Float
from pytorch3d.transforms import Transform3d
from torch import Tensor

from liblaf import cherries, melon
from liblaf.flame import FLAME, FlameConfig


class Config(cherries.BaseConfig):
    shape: Path = cherries.input("20-shape.npz")
    transform: Path = cherries.input("20-transform.npz")

    output: Path = cherries.output("30-expression.vtp")


def main(cfg: Config) -> None:
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    device: torch.device = torch.get_default_device()
    flame = FLAME(FlameConfig(batch_size=1))
    shape: Float[Tensor, " shape"] = torch.tensor(
        np.load(cfg.shape)["shape"], dtype=flame.dtype
    )
    transform: Float[Tensor, "4 4"] = torch.tensor(
        np.load(cfg.transform)["transform"], dtype=flame.dtype
    )
    transform3d: Transform3d = Transform3d(matrix=transform.T).to(device)

    expression: Float[Tensor, " expression"] = torch.zeros(
        (flame.config.expression_params,), dtype=flame.dtype
    )
    neutral_verts: Float[Tensor, "V 3"]
    neutral_verts, _ = flame(
        shape=shape[torch.newaxis], expression=expression[torch.newaxis]
    )
    neutral_verts = neutral_verts[0]
    neutral_verts = transform3d.transform_points(neutral_verts)
    mesh: pv.PolyData = pv.PolyData.from_regular_faces(
        neutral_verts.numpy(force=True), flame.faces
    )

    expressions: list[Float[Tensor, "V 3"]] = []
    expression = torch.zeros((flame.config.expression_params,), dtype=flame.dtype)
    expression[0] = 2.0
    expression[2] = 2.0
    expressions.append(expression.clone())
    expression = torch.zeros((flame.config.expression_params,), dtype=flame.dtype)
    expression[0] = -2.0
    expression[2] = 2.0
    expressions.append(expression.clone())
    expression = torch.zeros((flame.config.expression_params,), dtype=flame.dtype)
    expression[0] = 3.0
    expression[2] = 3.0
    expressions.append(expression.clone())
    expression = torch.zeros((flame.config.expression_params,), dtype=flame.dtype)
    expression[0] = -3.0
    expression[2] = 3.0
    expressions.append(expression.clone())
    for idx, expression in enumerate(expressions):
        verts: Float[Tensor, "vertices 3"]
        verts, _ = flame(
            shape=shape[torch.newaxis], expression=expression[torch.newaxis]
        )
        verts = verts[0]
        verts = transform3d.transform_points(verts)
        mesh.point_data[f"Expression{idx:03d}"] = (verts - neutral_verts).numpy(
            force=True
        )
    melon.save(mesh, cfg.output)


if __name__ == "__main__":
    cherries.main(main)
