import einx
import torch
import torch.nn as nn
from jaxtyping import Float


def random_masking_mae(x: torch.Tensor, mask_ratio: float):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence

    How to restore:
    return x_masked, mask, ids_restore
    use ids_restore to restore the original sequence

    mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


def random_masking_no_drop(
    x: torch.Tensor, mask_ratio: float, mask_token: Float[torch.Tensor, "1 1 D"]
):
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    if mask_token.ndim == 1:
        # shaped as (D,)
        mask_token = mask_token.unsqueeze(0).unsqueeze(0)
    assert mask_token.shape == (1, 1, D)

    # sort noise for each sample
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ids_masked = ids_shuffle[:, len_keep:]

    # set at
    x_masked = x.clone()
    einx.set_at(
        "b [l] c, b m, b m c -> b [l] c",
        x_masked,
        ids_masked,
        mask_token.repeat(N, ids_masked.shape[1], 1),
    )

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


if __name__ == "__main__":
    x = torch.randn(2, 8, 3)
    mask_token = torch.zeros(1, 1, 3)
    x_masked, mask, ids_restore = random_masking_no_drop(x, 0.5, mask_token)

    print(f"{x=}\n {mask=}\n {ids_restore=}\n {x_masked=}")

    # unshuffle
    # x_unshuffled = torch.gather(
    #     x_masked,
    #     dim=1,
    #     index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]),
    # )
    # print(x_unshuffled)
