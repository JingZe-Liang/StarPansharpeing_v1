import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _dist_rank() -> int:
    return dist.get_rank() if _dist_is_initialized() else 0


def _dist_world_size() -> int:
    return dist.get_world_size() if _dist_is_initialized() else 1


def _gather_batch_sizes(local_batch: int, device: torch.device) -> list[int]:
    local_size = torch.tensor([local_batch], device=device, dtype=torch.long)
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(_dist_world_size())]
    dist.all_gather(gathered_sizes, local_size)
    return [int(size.item()) for size in gathered_sizes]


def _pad_local_embeddings(x: Tensor, target_batch: int) -> Tensor:
    if x.shape[0] == target_batch:
        return x.contiguous()

    padded = x.new_zeros((target_batch, x.shape[1]))
    if x.shape[0] > 0:
        padded[: x.shape[0]] = x
    return padded.contiguous()


def _all_gather_detached_embeddings(x: Tensor) -> tuple[Tensor, int, int]:
    local_batch = int(x.shape[0])
    batch_sizes = _gather_batch_sizes(local_batch, x.device)
    global_batch = sum(batch_sizes)
    local_offset = sum(batch_sizes[: _dist_rank()])

    if global_batch == 0:
        return x.new_empty((0, x.shape[1])), global_batch, local_offset

    max_batch = max(batch_sizes)
    padded = _pad_local_embeddings(x.detach(), max_batch)
    gathered = [torch.empty_like(padded) for _ in batch_sizes]
    dist.all_gather(gathered, padded)

    valid_chunks = [
        chunk[:batch_size] for chunk, batch_size in zip(gathered, batch_sizes, strict=False) if batch_size > 0
    ]
    if len(valid_chunks) == 0:
        return x.new_empty((0, x.shape[1])), global_batch, local_offset
    return torch.cat(valid_chunks, dim=0), global_batch, local_offset


def _build_off_diagonal_mask(
    *,
    local_batch: int,
    global_batch: int,
    local_offset: int,
    device: torch.device,
) -> Tensor:
    mask = torch.ones((local_batch, global_batch), device=device, dtype=torch.bool)
    if local_batch == 0:
        return mask

    row_idx = torch.arange(local_batch, device=device)
    col_idx = local_offset + row_idx
    mask[row_idx, col_idx] = False
    return mask


class ARKDLoss(nn.Module):
    """
    Asymmetric relational knowledge distillation on summary embeddings.

    This module matches student and teacher batch geometry with a one-sided
    relational loss instead of forcing every pairwise distance to be identical.
    The implementation follows the ARKD formulation used in the paper summary
    embedded in this repository:

    1. Gather teacher and student summary embeddings across the global batch.
    2. Compute pairwise Euclidean distances ``D_t`` and ``D_s``.
    3. Normalize both distance matrices by the mean off-diagonal teacher
       distance, which makes the loss invariant to the absolute scale of the
       teacher embedding space.
    4. Split pairs with the median normalized teacher distance:
       pairs below the median are treated as close pairs, while the rest are
       treated as far pairs.
    5. Apply asymmetric one-sided penalties:
       ``relu(D_s - D_t)`` for close pairs, so student neighbors are only
       penalized when they drift too far apart;
       ``relu(D_t - D_s)`` for far pairs, so distant samples are only
       penalized when they collapse too much.
    6. Map the one-sided errors with Smooth L1 and average over all valid
       off-diagonal ordered pairs.

    The input tensors must be 2D and shaped as ``(batch, dim)``.
    ``teacher_summary`` is treated as a detached target, while gradients only
    flow through ``student_summary``.
    """

    def __init__(self, eps: float = 1e-6, gather_distributed: bool = True) -> None:
        """
        Args:
            eps: Lower bound for the teacher distance scale to avoid division
                by zero when teacher embeddings collapse.
            gather_distributed: Whether to gather summary embeddings from all
                distributed ranks before computing ARKD. The implementation
                supports uneven local batch sizes by padding before all-gather.
        """
        super().__init__()
        self.eps = eps
        self.gather_distributed = gather_distributed

    def _validate_inputs(self, student_summary: Tensor, teacher_summary: Tensor) -> None:
        if student_summary.ndim != 2:
            raise ValueError(f"student_summary must be 2D, got shape={tuple(student_summary.shape)}")
        if teacher_summary.ndim != 2:
            raise ValueError(f"teacher_summary must be 2D, got shape={tuple(teacher_summary.shape)}")
        if student_summary.shape != teacher_summary.shape:
            raise ValueError(
                "student_summary and teacher_summary must share the same shape, "
                f"got {tuple(student_summary.shape)} vs {tuple(teacher_summary.shape)}"
            )

    def _gather_global_columns(
        self, student_summary: Tensor, teacher_summary: Tensor
    ) -> tuple[Tensor, Tensor, int, int]:
        if not self.gather_distributed or not _dist_is_initialized() or _dist_world_size() == 1:
            global_batch = int(student_summary.shape[0])
            return student_summary.detach(), teacher_summary.detach(), global_batch, 0

        student_global, global_batch, local_offset = _all_gather_detached_embeddings(student_summary)
        teacher_global, _, _ = _all_gather_detached_embeddings(teacher_summary)
        return student_global, teacher_global, global_batch, local_offset

    def forward(self, student_summary: Tensor, teacher_summary: Tensor) -> Tensor:
        """
        Compute the ARKD loss for one local shard.

        Args:
            student_summary: Student summary embeddings with shape ``(B, D)``.
            teacher_summary: Teacher summary embeddings with shape ``(B, D)``.

        Returns:
            A scalar tensor. Under distributed training, each rank computes the
            contribution of its local rows against the gathered global columns,
            then rescales by world size so that gradient averaging matches the
            full global ordered-pair objective.
        """
        self._validate_inputs(student_summary, teacher_summary)

        student_rows = student_summary.to(dtype=torch.float32)
        teacher_rows = teacher_summary.detach().to(dtype=torch.float32)
        student_cols, teacher_cols, global_batch, local_offset = self._gather_global_columns(student_rows, teacher_rows)

        if global_batch < 2 or student_rows.shape[0] == 0:
            return student_rows.new_zeros(())

        pair_mask = _build_off_diagonal_mask(
            local_batch=int(student_rows.shape[0]),
            global_batch=global_batch,
            local_offset=local_offset,
            device=student_rows.device,
        )
        if not pair_mask.any():
            return student_rows.new_zeros(())

        teacher_dist = torch.cdist(teacher_rows, teacher_cols, p=2)
        student_dist = torch.cdist(student_rows, student_cols, p=2)

        teacher_valid = teacher_dist[pair_mask]
        teacher_scale = teacher_valid.mean().clamp_min(self.eps)

        teacher_dist = teacher_dist / teacher_scale
        student_dist = student_dist / teacher_scale
        median_dist = teacher_dist[pair_mask].median()

        close_mask = teacher_dist < median_dist
        far_mask = ~close_mask

        close_error = torch.relu(student_dist - teacher_dist)
        far_error = torch.relu(teacher_dist - student_dist)

        close_penalty = F.smooth_l1_loss(close_error, torch.zeros_like(close_error), reduction="none")
        far_penalty = F.smooth_l1_loss(far_error, torch.zeros_like(far_error), reduction="none")

        loss_matrix = close_mask * close_penalty + far_mask * far_penalty
        local_loss_sum = loss_matrix[pair_mask].sum()
        normalizer = float(global_batch * (global_batch - 1))
        return local_loss_sum * float(_dist_world_size()) / normalizer
