from pathlib import Path

import pooch
import pydantic
import wrapt
from environs import env


def _flame_model_pooch_factory() -> pooch.Pooch:
    return pooch.create(
        pooch.os_cache("liblaf/flame-pytorch"),
        "https://github.com/soubhiksanyal/RingNet/raw/master/flame_model/",
        registry={
            "flame_dynamic_embedding.npy": "sha256:fd046a4740f5e6837e65622d0e050273bb71bbcb1ad6ab322474013a2a23de37",
            "flame_static_embedding.pkl": "sha256:2881bdad8e61e87aad83a3df2e04655c534694089f68ca7144e7350da9e8ad62",
        },
    )


_flame_model_pooch: pooch.Pooch = wrapt.LazyObjectProxy(_flame_model_pooch_factory)  # pyright: ignore[reportAssignmentType]


def _default_static_landmark_embedding_path() -> Path:
    return Path(_flame_model_pooch.fetch("flame_static_embedding.pkl"))


def _default_dynamic_landmark_embedding_path() -> Path:
    return Path(_flame_model_pooch.fetch("flame_dynamic_embedding.npy"))


class FlameConfig(pydantic.BaseModel):
    flame_model_path: Path = pydantic.Field(
        default_factory=lambda: env.path(
            "FLAME_MODEL_PATH", Path("./model/generic_model.pkl")
        )
    )
    """flame model path"""

    static_landmark_embedding_path: Path = pydantic.Field(
        default_factory=_default_static_landmark_embedding_path
    )
    """Static landmark embeddings path for FLAME"""

    dynamic_landmark_embedding_path: Path = pydantic.Field(
        default_factory=_default_dynamic_landmark_embedding_path
    )
    """Dynamic contour embedding path for FLAME"""

    shape_params: int = 100
    """the number of shape parameters"""

    expression_params: int = 50
    """the number of expression parameters"""

    pose_params: int = 6
    """the number of pose parameters"""

    use_face_contour: bool = True
    """If true apply the landmark loss on also on the face contour."""

    use_3d_translation: bool = True
    """If true apply the landmark loss on also on the face contour."""

    optimize_eyeballpose: bool = True
    """If true optimize for the eyeball pose."""

    optimize_neckpose: bool = True
    """If true optimize for the neck pose."""

    num_worker: int = 4
    """pytorch number worker."""

    batch_size: int = 1
    """Training batch size."""

    ring_margin: float = 0.5
    """ring margin."""

    ring_loss_weight: float = 1.0
    """weight on ring loss."""

    @property
    def use_3D_translation(self) -> bool:  # noqa: N802
        return self.use_3d_translation

    @use_3D_translation.setter
    def use_3D_translation(self, value: bool) -> None:  # noqa: N802
        self.use_3d_translation = value


def get_config() -> FlameConfig:
    return FlameConfig()
