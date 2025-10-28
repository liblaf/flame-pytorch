import pytorch3d.io
import pytorch3d.loss
import pytorch3d.ops
import torch
from jaxtyping import Float, Integer
from pytorch3d.structures import Meshes
from torch import Tensor

from liblaf import grapes
from liblaf.flame_pytorch import FLAME, FlameConfig


def fitting(flame: FLAME, target: Meshes) -> Float[Tensor, "batch shape"]:
    shape: Float[Tensor, "batch shape"] = torch.zeros(
        (flame.batch_size, flame.config.shape_params), requires_grad=True
    )
    optimizer = torch.optim.LBFGS([shape])
    faces: Float[Tensor, "batch faces 3"] = torch.as_tensor(flame.faces)[torch.newaxis]
    flame.eval()

    def closure() -> Float[Tensor, ""]:
        optimizer.zero_grad()
        vertices, _ = flame(shape=shape)
        source = Meshes(vertices, faces)
        source_verts: Float[Tensor, "batch sample 3"] = (
            pytorch3d.ops.sample_points_from_meshes(source)
        )  # pyright: ignore[reportAssignmentType]
        # source_verts = source.verts_padded()
        target_verts: Float[Tensor, "batch sample 3"] = (
            pytorch3d.ops.sample_points_from_meshes(target)
        )  # pyright: ignore[reportAssignmentType]
        # target_verts = target.verts_padded()
        loss: Float[Tensor, ""]
        loss, _ = pytorch3d.loss.chamfer_distance(
            source_verts, target_verts, point_reduction="sum", single_directional=True
        )  # pyright: ignore[reportAssignmentType]
        loss.backward()
        return loss

    for _ in range(128):
        loss: Float[Tensor, ""] = optimizer.step(closure)
        ic(loss)
    return shape


def main() -> None:
    grapes.logging.init()
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    batch: int = 1
    flame = FLAME(FlameConfig(batch_size=batch))
    faces: Integer[Tensor, "batch faces 3"] = torch.as_tensor(
        flame.faces, dtype=torch.int32
    )[torch.newaxis]

    init_verts: Float[Tensor, "batch vertices 3"]
    init_verts, _ = flame(shape=torch.zeros((batch, flame.config.shape_params)))
    pytorch3d.io.save_ply("init.ply", init_verts[0], faces[0])  # pyright: ignore[reportArgumentType]

    shape_expected: Float[Tensor, "batch shape"] = torch.rand(
        (batch, flame.config.shape_params)
    )
    ic(shape_expected)
    target_verts: Float[Tensor, "batch vertices 3"]
    target_verts, _ = flame(shape=shape_expected)
    target: Meshes = Meshes(target_verts, faces).to(torch.get_default_device())
    pytorch3d.io.save_ply("target.ply", target_verts[0], faces[0])  # pyright: ignore[reportArgumentType]
    shape_actual: Float[Tensor, "batch shape"] = fitting(flame, target)
    ic(shape_actual)
    result_verts: Float[Tensor, "batch vertices 3"]
    result_verts, _ = flame(shape=shape_actual)
    pytorch3d.io.save_ply("result.ply", result_verts[0], faces[0])  # pyright: ignore[reportArgumentType]
    torch.testing.assert_close(shape_actual, shape_expected)


if __name__ == "__main__":
    main()
