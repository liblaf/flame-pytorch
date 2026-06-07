import logging

import attrs
import torch
import trimesh as tm
from jaxtyping import Float, Integer
from pytorch3d.transforms import Transform3d
from torch import Tensor
from torch.optim import LBFGS, Optimizer

from liblaf import cherries
from liblaf.flame_pytorch.flame import FLAME

logger: logging.Logger = logging.getLogger(__name__)


@attrs.frozen
class FitLandmarksWeights:
    landmarks: float = 1.0
    shape: float = 1e-3
    expression: float = 1e-3
    pose: float = 1e-2


@attrs.frozen
class FitLandmarksResult:
    transform: Transform3d
    shape: Float[Tensor, " S"]
    expression: Float[Tensor, " E"]
    pose: Float[Tensor, " P"]


def fit_landmarks(
    flame: FLAME,
    landmarks: Float[Tensor, "L 3"],
    landmark_indices: Integer[Tensor, " L"],
    weights: FitLandmarksWeights | None = None,
) -> FitLandmarksResult:
    device: torch.device = torch.get_default_device()
    if weights is None:
        weights: FitLandmarksWeights = FitLandmarksWeights()
    flame.eval()
    _verts, source = flame()
    matrix, _transformed, cost = tm.registration.procrustes(
        source[0, landmark_indices, :].numpy(force=True),
        landmarks.numpy(force=True),
    )
    logger.info("procrustes cost: %f", cost)
    transform: Transform3d = Transform3d(device=device, matrix=torch.tensor(matrix).mT)
    transform_inv: Transform3d = transform.inverse()
    matrix_inv: Float[Tensor, "4 4"] = torch.tensor(
        transform_inv.get_matrix(), requires_grad=True, device=device
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
    optimizer: Optimizer = LBFGS([matrix, shape, expression, pose])

    def closure() -> Float[Tensor, ""]:
        optimizer.zero_grad()
        _verts, source = flame(shape=shape, expression=expression, pose=pose)
        transform_inv: Transform3d = Transform3d(device=device, matrix=matrix_inv)
        target: Float[Tensor, "L 3"] = transform_inv.transform_points(landmarks)
        loss_landmarks: Float[Tensor, ""] = (
            (source[:, landmark_indices, :] - target).square().sum(dim=-1).mean()
        )
        loss_shape: Float[Tensor, ""] = shape.square().sum()
        loss_expression: Float[Tensor, ""] = expression.square().sum()
        loss_pose: Float[Tensor, ""] = pose.square().sum()
        loss: Float[Tensor, ""] = (
            weights.landmarks * loss_landmarks
            + weights.shape * loss_shape
            + weights.expression * loss_expression
            + weights.pose * loss_pose
        )
        cherries.log_metrics(
            {
                "fitting": {
                    "landmarks": {
                        "loss": {
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

    for step in range(100):
        cherries.set_step(step)
        optimizer.step(closure)

    transform_inv: Transform3d = Transform3d(device=device, matrix=matrix_inv)
    transform: Transform3d = transform_inv.inverse()
    return FitLandmarksResult(
        transform=transform, shape=shape, expression=expression, pose=pose
    )
