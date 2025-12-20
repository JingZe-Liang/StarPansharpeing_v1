import torch
from torch import Tensor


def _pixio_mae_mask(
    mask_ratio: float, grid: int, x: Tensor | None = None, bchw: tuple | None = None, device: str = "cuda"
):
    """
    pixio block masking type:
        mask the input image tokens by grids, not by pixels.

    ----
    Args:
        x: [N, L, D], Tensor
        mask_ratio: float between 0 and 1
        grid: int, grid size for block masking

    Return:
        x_masked: [N, L * (1 - mask_ratio), D], float Tensor
        mask: [N, L], Int Tensor, 0 is keep, 1 is remove
        ids_restore: [N, L], Tensor
    """
    assert x is not None or bchw is not None, "Either x or shape must be provided"
    if bchw is not None:
        N, D, H, W = bchw
    else:
        N, L, D = x.shape
        device = x.device
        H = W = int(L**0.5)
        x = x.view(N, H, W, D)

    num_patches = (H // grid) * (W // grid)
    len_keep = int(num_patches * (1 - mask_ratio))

    noise = torch.rand(N, num_patches, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    ids_masked = ids_shuffle[:, len_keep:]

    patch_grid = torch.arange(H * W, device=device).view(1, H, W)
    patch_grid = patch_grid.unfold(1, grid, grid).unfold(2, grid, grid)
    patch_grid = patch_grid.contiguous().view(1, -1, grid, grid)

    ids_keep_expanded = patch_grid[:, ids_keep].view(N, -1)
    ids_masked_expanded = patch_grid[:, ids_masked].view(N, -1)
    ids_restore = torch.cat((ids_keep_expanded, ids_masked_expanded), dim=1)
    ids_restore = torch.argsort(ids_restore, dim=1)

    if x is not None:
        x_masked = torch.gather(x.view(N, -1, D), dim=1, index=ids_keep_expanded.unsqueeze(-1).repeat(1, 1, D))
    else:
        x_masked = None

    mask = torch.ones([N, H * W], device=device)
    mask[:, : len_keep * (grid**2)] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_keep, ids_restore


def _kaiming_mae_mask(
    mask_ratio: float, x: torch.Tensor | None = None, bchw: tuple | None = None, device: str = "cuda"
):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    assert x is not None or bchw is not None, "Either x or shape must be provided"

    if bchw is not None:
        N, D, H, W = bchw
        L = H * W
    else:
        N, L, D = x.shape  # batch, length, dim
        device = x.device

    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(N, L, device=device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    if x is not None:
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    else:
        x_masked = None

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_keep, ids_restore


def mae_random_masking(
    mask_type: str,
    mask_ratio: float,
    grid: int,
    x: torch.Tensor | None = None,
    bchw: tuple | None = None,
):
    if mask_type == "pixio":
        return _pixio_mae_mask(mask_ratio, grid, x, bchw)
    elif mask_type == "kaiming":
        return _kaiming_mae_mask(mask_ratio, x, bchw)
    else:
        raise ValueError(f"Invalid mask type: {mask_type}")


def restore_sequence_by_ids(
    x: torch.Tensor,
    ids_restore: torch.Tensor,
    mask_token: torch.Tensor,
    num_prefixed_tokens: int = 1,
):
    mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] + num_prefixed_tokens - x.shape[1], 1)
    x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
    return x


if __name__ == "__main__":
    from einx import set_at, get_at

    x = torch.arange(1, 17).view(1, 16, 1).float().repeat_interleave(2, dim=0)
    mask_token = torch.ones(1, 1, 1) * -1

    # x_masked, mask, ids_restore = _pixio_mae_mask(x, mask_ratio=0.5, grid=2)
    x_masked, mask, ids_keep, ids_restore = _kaiming_mae_mask(mask_ratio=0.5, x=x, device="cpu")
    x_masked_einx = get_at("B [L] D, B L_masked -> B L_masked D", x, ids_keep)
    assert (x_masked == x_masked_einx).all(), "Masked x not equal!"
    print("x: ", x[0, :, 0])
    print("ids_keep: ", ids_keep)
    print("mask: ", mask[0, :].view(4, 4))
    print("x_masked:", x_masked[0, :, 0])
    print("x_masked_einx:", x_masked_einx[0, :, 0])
