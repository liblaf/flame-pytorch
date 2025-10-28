from pathlib import Path

import pyvista as pv
import torch
from jaxtyping import Float, Integer
from torch import Tensor

from liblaf import cherries, grapes, melon
from liblaf.flame_pytorch import FLAME, FlameConfig


class Config(cherries.BaseConfig):
    output: Path = cherries.output("10-neutral.obj")
    output_sparse: Path = cherries.output("10-neutral-sparse.obj")


def main(cfg: Config) -> None:
    grapes.logging.init()
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    batch: int = 1
    flame = FLAME(FlameConfig(batch_size=batch))
    faces: Integer[Tensor, "batch faces 3"] = torch.as_tensor(
        flame.faces, dtype=torch.int32
    )[torch.newaxis]

    verts: Float[Tensor, "batch vertices 3"]
    landmarks: Float[Tensor, "batch landmarks 3"]
    verts, landmarks = flame()
    mesh: pv.PolyData = pv.PolyData.from_regular_faces(
        verts[0].numpy(force=True), faces[0].numpy(force=True)
    )
    melon.io.save(cfg.output, mesh)
    melon.io.save_landmarks(cfg.output, landmarks[0].numpy(force=True))

    landmark_idx: Integer[Tensor, " landmark_idx"] = (
        torch.as_tensor([9, 28, 31, 32, 34, 36, 37, 40, 43, 46, 49, 52, 58, 65]) - 1
    )
    landmarks_partial: Float[Tensor, "batch landmarks 3"] = landmarks[
        0, landmark_idx, :
    ]
    melon.io.save(cfg.output_sparse, mesh)
    melon.io.save_landmarks(cfg.output_sparse, landmarks_partial.numpy(force=True))


if __name__ == "__main__":
    cherries.run(main)
