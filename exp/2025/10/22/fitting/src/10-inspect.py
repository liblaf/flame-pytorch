from pathlib import Path

import pyvista as pv
import torch
from jaxtyping import Float, Integer
from torch import Tensor

from liblaf import cherries, melon
from liblaf.flame_pytorch import FLAME, FlameConfig


class Config(cherries.BaseConfig):
    output: Path = cherries.output("10-neutral.obj")
    output_sparse: Path = cherries.output("10-neutral-sparse.obj")


def main(cfg: Config) -> None:
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    batch: int = 1
    flame = FLAME(FlameConfig(batch_size=batch))

    verts, landmarks = flame()
    mesh: pv.PolyData = pv.make_tri_mesh(verts[0].numpy(force=True), flame.faces)
    melon.io.save(mesh, cfg.output)
    melon.io.save_landmarks(landmarks[0].numpy(force=True), cfg.output)

    landmark_idx: Integer[Tensor, " L"] = (
        torch.tensor([9, 28, 31, 32, 34, 36, 37, 40, 43, 46, 49, 52, 58, 65]) - 1
    )
    landmarks_sparse: Float[Tensor, "B L 3"] = landmarks[:, landmark_idx, :]
    melon.io.save(mesh, cfg.output_sparse)
    melon.io.save_landmarks(landmarks_sparse[0].numpy(force=True), cfg.output_sparse)


if __name__ == "__main__":
    cherries.main(main)
