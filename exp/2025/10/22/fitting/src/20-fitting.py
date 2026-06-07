import logging
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from jaxtyping import Float, Integer
from pytorch3d.structures import Meshes
from torch import Tensor

from liblaf import cherries, melon
from liblaf.flame_pytorch import FLAME, FitScanResult, FlameConfig, fit_scan

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    target: Path = cherries.input("00-target.vtp")
    output: Path = cherries.output("20-shape.ply")
    output_params: Path = cherries.output("20-shape.npz")


FACE_GROUPS: list[str] = [
    "Chin",
    "EyelidBottom",
    "EyelidOuterBottom",
    "EyelidOuterTop",
    "EyelidTop",
    "Face",
    "HeadBack",
    "LipBottom",
    "LipOuterBottom",
    "LipOuterTop",
    "LipTop",
]
LANDMARK_INDICES: Integer[Tensor, "L 3"] = (
    torch.tensor(
        [9, 28, 31, 32, 34, 36, 37, 40, 43, 46, 49, 52, 58, 65], dtype=torch.int32
    )
    - 1
)


def prepare_target(target: pv.PolyData) -> Meshes:
    device: torch.device = torch.get_default_device()
    target: pv.PolyData = melon.tri.extract_groups(target, FACE_GROUPS)
    verts: Float[Tensor, "V 3"] = torch.tensor(
        target.points, dtype=torch.get_default_dtype()
    )
    faces: Integer[Tensor, "F 3"] = torch.tensor(
        target.regular_faces, dtype=torch.int32
    )
    target: Meshes = Meshes([verts], [faces]).to(device)
    return target


def main(cfg: Config) -> None:
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    target_pv: pv.PolyData = melon.io.load_polydata(cfg.target)
    target_landmarks: Float[Tensor, "L 3"] = torch.tensor(
        melon.io.load_landmarks(cfg.target), dtype=torch.get_default_dtype()
    )
    target: Meshes = prepare_target(target_pv)
    flame = FLAME(FlameConfig(batch_size=1))
    result: FitScanResult = fit_scan(
        flame=flame,
        target=target,
        landmarks=target_landmarks,
        landmark_indices=LANDMARK_INDICES,
    )
    matrix: Float[Tensor, "4 4"] = result.transform.get_matrix()[0].mT
    np.savez_compressed(
        cfg.output_params,
        allow_pickle=False,
        transform=matrix.numpy(force=True),
        shape=result.shape.numpy(force=True),
        expression=result.expression.numpy(force=True),
        pose=result.pose.numpy(force=True),
    )
    verts, _ = flame(
        shape=result.shape[torch.newaxis],
        expression=result.expression[torch.newaxis],
        pose=result.pose[torch.newaxis],
    )
    verts: Float[Tensor, "B V 3"] = result.transform.transform_points(verts)
    result: pv.PolyData = pv.make_tri_mesh(verts[0].numpy(force=True), flame.faces)
    melon.save(result, cfg.output)


if __name__ == "__main__":
    cherries.main(main)
