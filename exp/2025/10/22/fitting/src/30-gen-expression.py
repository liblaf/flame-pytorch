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
    params: Path = cherries.input("20-params.npz")
    output: Path = cherries.output("30-expression.vtp")


def main(cfg: Config) -> None:
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    device: torch.device = torch.get_default_device()
    dtype: torch.dtype = torch.get_default_dtype()
    flame = FLAME(FlameConfig(batch_size=1))

    params: dict[str, np.ndarray] = np.load(cfg.params)
    transform: Float[Tensor, "4 4"] = torch.tensor(params["transform"], dtype=dtype)
    transform3d: Transform3d = Transform3d(device=device, matrix=transform.mT)
    shape: Float[Tensor, " S"] = torch.tensor(params["shape"], dtype=dtype)
    pose: Float[Tensor, " P"] = torch.tensor(params["pose"], dtype=dtype)

    expression: Float[Tensor, " E"] = torch.zeros((flame.config.expression_params,))
    neutral_verts: Float[Tensor, "V 3"]
    neutral_verts, _ = flame(
        shape=shape.unsqueeze(0),
        expression=expression.unsqueeze(0),
        pose=pose.unsqueeze(0),
    )
    neutral_verts: Float[Tensor, "V 3"] = neutral_verts.squeeze(0)
    neutral_verts: Float[Tensor, "V 3"] = transform3d.transform_points(neutral_verts)
    mesh: pv.PolyData = pv.PolyData.from_regular_faces(
        neutral_verts.numpy(force=True), flame.faces
    )

    expressions: list[Float[Tensor, "V 3"]] = []
    expression: Float[Tensor, " E"] = torch.zeros((flame.config.expression_params,))
    expression[0] = 2.0
    expression[2] = 2.0
    expressions.append(expression)
    expression: Float[Tensor, " E"] = torch.zeros((flame.config.expression_params,))
    expression[0] = -2.0
    expression[2] = 2.0
    expressions.append(expression)
    expression: Float[Tensor, " E"] = torch.zeros((flame.config.expression_params,))
    expression[0] = 3.0
    expression[2] = 3.0
    expressions.append(expression)
    expression: Float[Tensor, " E"] = torch.zeros((flame.config.expression_params,))
    expression[0] = -3.0
    expression[2] = 3.0
    expressions.append(expression)
    for idx, expression in enumerate(expressions):
        verts, _ = flame(
            shape=shape.unsqueeze(0),
            expression=expression.unsqueeze(0),
            pose=pose.unsqueeze(0),
        )
        verts: Float[Tensor, "V 3"] = verts.squeeze(0)
        verts: Float[Tensor, "V 3"] = transform3d.transform_points(verts)
        mesh.point_data[f"Expression{idx:03d}"] = (verts - neutral_verts).numpy(
            force=True
        )
    melon.save(mesh, cfg.output)


if __name__ == "__main__":
    cherries.main(main)
