from pathlib import Path

import attrs
import numpy as np
import pyvista as pv
import torch
import trimesh as tm
from jaxtyping import Float, Integer
from loguru import logger
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d
from torch import Tensor

from liblaf import cherries, melon
from liblaf.flame_pytorch import FLAME, FlameConfig


@attrs.define
class Loss:
    chamfer: float = 1.0
    chamfer_normals: float = 0.0
    landmarks: float = 1.0
    regularization: float = 1e-3

    def __call__(
        self,
        shape: Float[Tensor, " shape"],
        source: Meshes,
        target: Meshes,
        source_landmarks: Float[Tensor, "batch landmarks 3"],
        target_landmarks: Float[Tensor, "batch landmarks 3"],
    ) -> Float[Tensor, ""]:
        loss_chamfer: Float[Tensor, ""]
        loss_chamfer_normals: Float[Tensor, ""]
        loss_chamfer, loss_chamfer_normals = self.loss_chamfer(source, target)
        loss_landmarks: Float[Tensor, ""] = self.loss_landmarks(
            source_landmarks, target_landmarks
        )
        loss_regularization: Float[Tensor, ""] = self.loss_regularization(shape)
        loss: Float[Tensor, ""] = (
            loss_chamfer + loss_chamfer_normals + loss_landmarks + loss_regularization
        )
        cherries.log_metrics(
            {
                "loss": {
                    "chamfer": loss_chamfer.item(),
                    "chamfer_normals": loss_chamfer_normals.item(),
                    "landmarks": loss_landmarks.item(),
                    "regularization": loss_regularization.item(),
                    "total": loss.item(),
                }
            }
        )
        return loss

    def loss_chamfer(
        self, source: Meshes, target: Meshes
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
        source_points: Float[Tensor, "batch samples 3"]
        source_normals: Float[Tensor, "batch samples 3"]
        source_points, source_normals = sample_points_from_meshes(  # pyright: ignore[reportAssignmentType]
            source, return_normals=True
        )
        target_points: Float[Tensor, "batch samples 3"]
        target_normals: Float[Tensor, "batch samples 3"]
        target_points, target_normals = sample_points_from_meshes(  # pyright: ignore[reportAssignmentType]
            target, return_normals=True
        )
        chamfer: Float[Tensor, ""]
        chamfer_normals: Float[Tensor, ""]
        chamfer, chamfer_normals = chamfer_distance(
            target_points,
            source_points,
            x_normals=source_normals,
            y_normals=target_normals,
            single_directional=True,
        )  # pyright: ignore[reportAssignmentType]
        return self.chamfer * chamfer, self.chamfer_normals * chamfer_normals

    def loss_landmarks(
        self,
        source_landmarks: Float[Tensor, "batch landmarks 3"],
        target_landmarks: Float[Tensor, "batch landmarks 3"],
    ) -> Float[Tensor, ""]:
        loss_landmarks: Float[Tensor, ""] = torch.mean(
            torch.sum(torch.square(source_landmarks - target_landmarks), dim=-1)
        )
        return self.landmarks * loss_landmarks

    def loss_regularization(self, shape: Float[Tensor, " shape"]) -> Float[Tensor, ""]:
        regularization: Float[Tensor, ""] = torch.sum(shape**2)
        return self.regularization * regularization


def landmark_indices() -> Integer[Tensor, " landmarks"]:
    return torch.as_tensor([9, 28, 31, 32, 34, 36, 37, 40, 43, 46, 49, 52, 58, 65]) - 1


def fit_transform(
    flame: FLAME, target_landmarks: Float[Tensor, "landmarks 3"]
) -> Float[Tensor, "4 4"]:
    flame.eval()
    source_landmarks: Float[Tensor, "landmarks 3"]
    _, source_landmarks = flame()
    source_landmarks = source_landmarks[0, landmark_indices(), :]
    matrix: Float[np.ndarray, "4 4"]
    cost: float
    matrix, _, cost = tm.registration.procrustes(
        source_landmarks.numpy(force=True), target_landmarks.numpy(force=True)
    )
    logger.info("Procrustes cost: {}", cost)
    return torch.as_tensor(matrix, dtype=torch.float32)


def fit_shape(
    flame: FLAME,
    loss_fn: Loss,
    target: Meshes,
    target_landmarks: Float[Tensor, "batch landmarks 3"],
    transform: Float[Tensor, "#batch 4 4"],
    shape: Float[Tensor, "batch shape"] | None = None,
    *,
    fit_transform: bool = True,
) -> tuple[Float[Tensor, "#batch 4 4"], Float[Tensor, "batch shape"]]:
    device: torch.device = torch.get_default_device()
    flame.eval()
    faces: Integer[Tensor, "#batch faces 3"] = torch.as_tensor(
        flame.faces, dtype=torch.int32
    )[torch.newaxis]
    if shape is None:
        shape = torch.zeros(
            (flame.config.batch_size, flame.config.shape_params), requires_grad=True
        )
    transform = torch.tensor(transform, dtype=torch.float32, requires_grad=True)
    params: list[Tensor] = [shape]
    if fit_transform:
        params.append(transform)
    optimizer = torch.optim.LBFGS(params)

    def closure() -> Float[Tensor, ""]:
        optimizer.zero_grad()
        verts: Float[Tensor, "batch vertices 3"]
        landmarks: Float[Tensor, "batch landmarks 3"]
        verts, landmarks = flame(shape=shape)
        landmarks = landmarks[:, landmark_indices(), :]
        transform3d: Transform3d = Transform3d(matrix=transform.T).to(device)
        verts = transform3d.transform_points(verts)
        landmarks = transform3d.transform_points(landmarks)
        source: Meshes = Meshes(verts, faces).to(device)
        loss: Float[Tensor, ""] = loss_fn(
            shape=shape,
            source=source,
            target=target,
            source_landmarks=landmarks,
            target_landmarks=target_landmarks,
        )
        loss.backward()
        return loss

    for step in range(128):
        loss: Float[Tensor, ""] = optimizer.step(closure)
        cherries.log_metric("loss", loss, step=step)
        ic(shape[0, :5], transform)
    return shape, transform


class Config(cherries.BaseConfig):
    output: Path = cherries.output("20-shape.ply")
    target: Path = cherries.input("00-target.vtp")


def main(cfg: Config) -> None:
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    device: torch.device = torch.get_default_device()
    target_pv: pv.PolyData = melon.io.load_polydata(cfg.target)
    target_pv = melon.tri.extract_groups(
        target_pv,
        [
            "Ear",
            "EarNeckBack",
            "EarSocket EyeSocketTop",
            "HeadBack",
            "LipInnerBottom",
            "LipInnerTop",
            "MouthSocketBottom",
            "MouthSocketTop",
            "NeckBack",
            "NeckFront",
        ],
        invert=True,
    )
    target_pv.triangulate(inplace=True)
    target: Meshes = Meshes(
        verts=torch.as_tensor(target_pv.points, dtype=torch.float32)[torch.newaxis],
        faces=torch.as_tensor(target_pv.regular_faces, dtype=torch.int32)[
            torch.newaxis
        ],
    ).to(device)
    target_landmarks: Float[Tensor, "landmarks 3"] = torch.as_tensor(
        melon.io.load_landmarks(cfg.target)
    )

    flame = FLAME(FlameConfig(batch_size=1))
    transform: Float[Tensor, "4 4"] = fit_transform(flame, target_landmarks)
    ic(transform)

    shape: Float[Tensor, "batch shape"]
    shape, transform = fit_shape(
        flame,
        loss_fn=Loss(),
        target=target,
        target_landmarks=target_landmarks,
        transform=transform,
        fit_transform=False,
    )
    # shape, transform = fit_shape(
    #     flame,
    #     loss_fn=Loss(landmarks=0.0),
    #     target=target,
    #     target_landmarks=target_landmarks,
    #     transform=transform,
    #     shape=shape,
    # )
    transform3d: Transform3d = Transform3d(matrix=transform.T).to(device)
    verts: Float[Tensor, "batch vertices 3"]
    verts, _ = flame(shape=shape)
    verts = transform3d.transform_points(verts)
    result: pv.PolyData = pv.PolyData.from_regular_faces(
        verts[0].numpy(force=True), flame.faces
    )
    melon.io.save(cfg.output, result)


if __name__ == "__main__":
    cherries.run(main)
