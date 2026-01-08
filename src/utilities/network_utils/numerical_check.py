import torch


def raise_if_nonfinite_tensor(t: torch.Tensor, *, tensor_name: str, model: torch.nn.Module) -> None:
    """
    Check the network training numerical stability.
    If `t` a tensor output from the network is nan/inf, check the model
    param has any nan or inf.
    """
    if torch.isfinite(t).all():
        return

    with torch.no_grad():
        finite_mask = torch.isfinite(t)
        n_total = int(t.numel())
        n_finite = int(finite_mask.sum().item())
        n_nonfinite = n_total - n_finite

        t_fp32 = t.detach().to(dtype=torch.float32)
        valid_values = t_fp32[~torch.isnan(t_fp32)]
        if valid_values.numel() > 0:
            t_min = valid_values.min().item()
            t_max = valid_values.max().item()
        else:
            t_min = "all nan"
            t_max = "all nan"

        bad_params: list[str] = []
        for name, p in model.named_parameters():
            if not p.is_floating_point():
                continue
            if not p.requires_grad:
                continue
            if not torch.isfinite(p).all():
                bad_params.append(f"{name} dtype={p.dtype} shape={tuple(p.shape)}")
                if len(bad_params) >= 30:
                    break

    raise RuntimeError(
        f"[NonFinite] {tensor_name} contains NaN/Inf: "
        f"nonfinite={n_nonfinite}/{n_total}, dtype={t.dtype}, shape={tuple(t.shape)}, "
        f"nanmin={t_min}, nanmax={t_max}. "
        f"Non-finite params (first {len(bad_params)}): {bad_params}"
    )


def raise_if_nonfinite_params(self, model: torch.nn.Module, *, model_name: str, max_items: int = 30) -> None:
    """Raise if the model has any nan/inf param."""
    bad_params: list[str] = []
    for name, p in model.named_parameters():
        if not p.is_floating_point():
            continue
        if not p.requires_grad:
            continue
        if torch.isfinite(p).all():
            continue
        bad_params.append(f"{name} dtype={p.dtype} shape={tuple(p.shape)}")
        if max_items is not None and len(bad_params) >= max_items:
            break

    if bad_params:
        raise RuntimeError(f"[NonFinite] {model_name} has non-finite params: {bad_params}")
