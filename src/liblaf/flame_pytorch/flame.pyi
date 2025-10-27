import numpy as np
from jaxtyping import Float, Integer
from torch import Tensor, nn

from .config import FlameConfig

class FLAME(nn.Module):
    config: FlameConfig
    batch_size: int
    use_face_contour: bool
    use_3D_translation: bool  # noqa: N815

    faces: Integer[np.ndarray, "faces 3"]

    def __init__(self, config: FlameConfig | None = None) -> None: ...
    def __call__(
        self,
        shape: Float[Tensor, "batch shape"] | None = None,
        expression: Float[Tensor, "batch expression"] | None = None,
        pose: Float[Tensor, "batch pose"] | None = None,
        neck_pose: Float[Tensor, "batch 3"] | None = None,
        eye_pose: Float[Tensor, "batch 6"] | None = None,
        translation: Float[Tensor, "batch 3"] | None = None,
    ) -> tuple[Float[Tensor, "batch vertices"], Float[Tensor, "batch landmarks 3"]]: ...
    def forward(
        self,
        shape: Float[Tensor, "batch shape"] | None = None,
        expression: Float[Tensor, "batch expression"] | None = None,
        pose: Float[Tensor, "batch pose"] | None = None,
        neck_pose: Float[Tensor, "batch 3"] | None = None,
        eye_pose: Float[Tensor, "batch 6"] | None = None,
        translation: Float[Tensor, "batch 3"] | None = None,
    ) -> tuple[Float[Tensor, "batch vertices"], Float[Tensor, "batch landmarks 3"]]: ...
