import attrs
import torch
from jaxtyping import Float
from pytorch3d.structures import Meshes
from torch import Tensor

from .flame import FLAME


@attrs.define
class Weights:
    chamfer: float = 1.0
    chamfer_normals: float = 0.0
    landmarks: float = 1.0
    regularization: float = 1e-6


@attrs.define
class Fitting:
    optimizer: torch.optim.Optimizer
    shape: Float[Tensor, "batch shape"]
    target: Meshes
    transform: Float[Tensor, "#batch 4 4"]
    flame: FLAME = attrs.field(factory=FLAME)
    max_iter: int = 128
    target_landmarks: Float[Tensor, "batch landmark"] | None = None
    weights: Weights = attrs.field(factory=Weights)

    def __init__(
        self,
        flame: FLAME,
        *,
        shape: Float[Tensor, "batch shape"] | bool = True,
        transform: Float[Tensor, "#batch 4 4"] | bool = True,
    ) -> None:
        params: list[Tensor] = []
        if shape is True:
            shape = torch.zeros((flame.config.shape_params,), requires_grad=True)
        if shape is not False:
            params.append(shape)
        if transform is True:
            transform = torch.eye(4, requires_grad=True)
        if transform is not False:
            params.append(transform)
        self.optimizer = torch.optim.LBFGS(params)

    def fit(
        self,
        target: Meshes,
        target_landmarks: Float[Tensor, "batch landmark"] | None = None,
    ) -> Meshes:
        return target

    def loss_landmarks(self) -> Float[Tensor, ""]:
        return torch.tensor(0.0)
