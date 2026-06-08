import logging
from typing import cast

import attrs
import torch
import trimesh as tm
from jaxtyping import Float, Integer
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops.sample_points_from_meshes import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d
from torch import Tensor
from torch.optim import LBFGS, Optimizer

from liblaf import cherries
from liblaf.flame_pytorch.flame import FLAME

logger: logging.Logger = logging.getLogger(__name__)


@attrs.frozen
class FitScanWeights:
    chamfer: float = 2.0
    landmarks: float = 1e-2
    shape: float = 1e-4
    expression: float = 1e-4
    pose: float = 1e-3


@attrs.frozen
class FitScanResult:
    transform: Transform3d
    shape: Float[Tensor, " S"]
    expression: Float[Tensor, " E"]
    pose: Float[Tensor, " P"]


def fit_scan(
    flame: FLAME,
    target: Meshes,
    landmarks: Float[Tensor, "L 3"],
    landmark_indices: Integer[Tensor, " L"],
    weights: FitScanWeights | None = None,
) -> FitScanResult:
    device: torch.device = torch.get_default_device()
    if weights is None:
        weights: FitScanWeights = FitScanWeights()
    flame.eval()
    _verts, source_landmarks = flame()
    matrix_inv, _transformed, cost = tm.registration.procrustes(
        landmarks.numpy(force=True),
        source_landmarks.squeeze(0)[landmark_indices].numpy(force=True),
    )
    logger.info("procrustes cost: %f", cost)

    matrix_inv: Float[Tensor, "4 4"] = torch.tensor(
        matrix_inv, dtype=torch.get_default_dtype(), requires_grad=True
    )
    shape: Float[Tensor, " S"] = torch.zeros(
        (flame.config.shape_params,), requires_grad=True
    )
    expression: Float[Tensor, " E"] = torch.zeros(
        (flame.config.expression_params,), requires_grad=True
    )
    pose: Float[Tensor, " P"] = torch.zeros(
        (flame.config.pose_params,), requires_grad=True
    )

    def closure() -> Float[Tensor, ""]:
        optimizer.zero_grad()
        source_verts, source_landmarks = flame(
            shape=shape.unsqueeze(0),
            expression=expression.unsqueeze(0),
            pose=pose.unsqueeze(0),
        )
        source_mesh: Meshes = Meshes(
            source_verts, torch.tensor(flame.faces).unsqueeze(0)
        ).to(device)
        source_landmarks: Float[Tensor, "L 3"] = source_landmarks.squeeze(0)[
            landmark_indices
        ]
        transform_inv: Transform3d = Transform3d(device=device, matrix=matrix_inv.mT)
        target_verts: Float[Tensor, "V 3"] = transform_inv.transform_points(
            target.verts_list()[0]
        )
        target_mesh: Meshes = Meshes([target_verts], target.faces_list()).to(device)
        target_landmarks: Float[Tensor, "L 3"] = transform_inv.transform_points(
            landmarks
        )

        source_points: Float[Tensor, "1 P 3"] = cast(
            "Tensor", sample_points_from_meshes(source_mesh)
        )
        target_points: Float[Tensor, "1 P 3"] = cast(
            "Tensor", sample_points_from_meshes(target_mesh)
        )
        loss_chamfer, _loss_normals = chamfer_distance(
            target_points, source_points, single_directional=True
        )

        loss_landmarks: Float[Tensor, ""] = (
            (source_landmarks - target_landmarks).square().sum(dim=-1).mean()
        )

        loss_shape: Float[Tensor, ""] = shape.square().mean()
        loss_expression: Float[Tensor, ""] = expression.square().mean()
        loss_pose: Float[Tensor, ""] = pose.square().mean()
        loss: Float[Tensor, ""] = (
            weights.chamfer * loss_chamfer
            + weights.landmarks * loss_landmarks
            + weights.shape * loss_shape
            + weights.expression * loss_expression
            + weights.pose * loss_pose
        )
        cherries.log_metrics(
            {
                "fitting": {
                    "landmarks": {
                        "loss": {
                            "chamfer": loss_chamfer,
                            "landmarks": loss_landmarks,
                            "shape": loss_shape,
                            "expression": loss_expression,
                            "pose": loss_pose,
                            "total": loss,
                        }
                    }
                }
            }
        )
        loss.backward()
        return loss

    optimizer: Optimizer = LBFGS(
        [matrix_inv, shape, expression, pose],
        tolerance_grad=0.0,
        tolerance_change=0.0,
        line_search_fn="strong_wolfe",
    )
    for step in range(100):
        cherries.set_step(step)
        optimizer.step(closure)

    transform_inv: Transform3d = Transform3d(device=device, matrix=matrix_inv.mT)
    transform: Transform3d = transform_inv.inverse()
    return FitScanResult(
        transform=transform, shape=shape, expression=expression, pose=pose
    )
