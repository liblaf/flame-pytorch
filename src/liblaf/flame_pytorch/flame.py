import torch
from jaxtyping import Float
from torch import Tensor

from .config import FlameConfig
from .upstream import FLAME as _FLAME


class FLAME(_FLAME):
    config: FlameConfig

    shapedirs: Tensor

    def __init__(self, config: FlameConfig | None = None) -> None:
        if config is None:
            config = FlameConfig()
        super().__init__(config)
        self.config = config
        if torch.cuda.is_available():
            self.cuda()

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        shape: Float[Tensor, "#batch shape"] | None = None,
        expression: Float[Tensor, "#batch expression"] | None = None,
        pose: Float[Tensor, "#batch pose"] | None = None,
        neck_pose: Float[Tensor, "#batch 3"] | None = None,
        eye_pose: Float[Tensor, "#batch 6"] | None = None,
        translation: Float[Tensor, "#batch 3"] | None = None,
    ) -> tuple[Float[Tensor, "#batch vertices 3"], Float[Tensor, "#batch landmarks 3"]]:
        if shape is None:
            shape = torch.zeros(
                (self.config.batch_size, self.config.shape_params),
                device=self.shapedirs.device,
                requires_grad=False,
            )
        if expression is None:
            expression = torch.zeros(
                (self.config.batch_size, self.config.expression_params),
                device=self.shapedirs.device,
                requires_grad=False,
            )
        if pose is None:
            pose = torch.zeros(
                (self.config.batch_size, self.config.pose_params),
                device=self.shapedirs.device,
                requires_grad=False,
            )
        return super().forward(
            shape_params=shape,
            expression_params=expression,
            pose_params=pose,
            neck_pose=neck_pose,
            eye_pose=eye_pose,
            transl=translation,
        )
