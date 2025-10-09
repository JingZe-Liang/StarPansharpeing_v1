import torch
from accelerate.state import PartialState
from torch.distributed.tensor import DTensor, Replicate, Shard

from ..logging import log_print


def to_full_tensor(tensor: DTensor) -> torch.Tensor:
    """
    Convert a DTensor to a full tensor by replicating it across all processes.

    Args:
        tensor: Input DTensor to convert

    Returns:
        torch.Tensor: Full tensor (not DTensor)
    """

    if not isinstance(tensor, DTensor):
        return tensor

    state = PartialState()
    device = state.device

    if tensor._local_tensor.device != device:
        tensor._local_tensor = tensor._local_tensor.to(device)

    try:
        return tensor.full_tensor()
    except Exception as e:
        log_print(f"full_tensor() failed: {e}", level="warning", warn_once=True)
        try:
            replicated = tensor.redistribute(placements=[Replicate()])
            return replicated.to_local()
        except Exception as e2:
            log_print(f"redistribute() failed: {e2}", level="warning", warn_once=True)
            log_print(
                "Failed to get full tensor, using local tensor instead. "
                "Results may be incomplete in distributed training.",
                level="warning",
                warn_once=True,
            )
            return tensor._local_tensor.to(device)


def safe_dtensor_operation(
    tensor: DTensor | torch.Tensor, prefer_full: bool = True
) -> torch.Tensor:
    if not isinstance(tensor, DTensor):
        return tensor

    device = PartialState().device
    if tensor._local_tensor.device != device:
        tensor._local_tensor = tensor._local_tensor.to(device)

    if prefer_full:
        return to_full_tensor(tensor)
    else:
        return tensor.to_local()


def get_tensor_info(tensor: DTensor | torch.Tensor) -> dict:
    if isinstance(tensor, DTensor):
        return {
            "type": "DTensor",
            "global_shape": tensor.shape,
            "local_shape": tensor._local_tensor.shape,
            "device": tensor._local_tensor.device,
            "placements": str(tensor.placements),
            "device_mesh": str(tensor.device_mesh),
        }
    else:
        return {
            "type": "Tensor",
            "shape": tensor.shape,
            "device": tensor.device,
            "dtype": tensor.dtype,
        }
