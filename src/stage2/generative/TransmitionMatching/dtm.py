from typing import cast

import torch
from einops import rearrange
from torch import Tensor, nn
from torchdiffeq import odeint


class DTM:
    def __init__(self, backbone_disc_T: int = 8, fm_head_N: int = 25):
        self.backbone_disc_T = backbone_disc_T
        self.fm_head_N = fm_head_N

        assert self.backbone_disc_T > 0 and self.fm_head_N > 0, "backbone_disc_T and fm_head_N must be positive"

    def loss(
        self,
        backbone: nn.Module,  # Denoted as `f^\theta`
        head: nn.Module,  # Denoted as `g^\theta`
        patch_size: int,  # Patch size
        X_T: Tensor,  # Image from training set `X_T~p_T`
        conditions: Tensor | None = None,  # Optional conditioning
        to_1d: bool = True,
    ) -> dict:
        # Convert image to sequence using patchify
        if to_1d:
            X_T = rearrange(
                X_T,
                "b c (h dh) (w dw) -> b (h w) (dh dw c)",
                dh=patch_size,
                dw=patch_size,
            )

        bsz, seq_len = X_T.shape[:2]

        # Sample time step `t~U[T-1]`
        T = self.backbone_disc_T
        t = torch.randint(0, T, (bsz,))

        # Sample a pair `(X_t,Y)~q_{t,Y|T}(.|X_T)``
        X_0 = torch.rand_like(X_T)
        X_t = (1 - t / T).view(-1, 1, 1) * X_0 + (t / T).view(-1, 1, 1) * X_T
        Y = X_T - X_0

        # Backbone forward
        h_t = backbone(X_t, t, conditions)

        # Reshape sequence for head
        h_t = h_t.view(bsz * seq_len, -1)
        Y = Y.view(bsz * seq_len, -1)
        t = t.repeat_interleave(seq_len)

        # Flow matching loss with the head as velocity and Y as target
        Y_0 = torch.rand_like(Y)
        s = torch.rand(bsz * seq_len)
        Y_s = (1 - s).view(-1, 1) * Y_0 + s.view(-1, 1) * Y

        # Head forward
        u = head(h_t, t, Y_s, s)
        loss = torch.nn.functional.mse_loss(u, Y - Y_0)

        # To predicted Y
        X_pred = u + Y_0 + X_0
        if to_1d:
            X_pred = rearrange(
                X_pred,
                "b (h w) (dh dw c) -> b c (h dh) (w dw)",
                dh=patch_size,
                dw=patch_size,
            )

        return {
            "loss": loss,
            "h_t": h_t,
            "pred_u": u,
            "pred_X": X_pred,
            "Y_s": Y_s,
            "t_s": [t, s],
        }

    def sample_head(
        self,
        head: nn.Module,
        h_t: Tensor,
        t: Tensor,
        solver: str = "euler",
        steps: int | None = None,
        verbose: bool = False,
    ):
        if steps is None:
            steps = self.fm_head_N

        # Start from Gaussian
        Y_0 = torch.randn_like(h_t)

        # Prefix head's inputs
        call_fn = lambda Y_s, s: head(h_t, t, Y_s, s)

        # Time grid
        Ns = torch.linspace(0, 1, steps + 1, device=h_t.device)
        Y_N = odeint(call_fn, Y_0, Ns, method=solver)
        Y_N = cast(Tensor, Y_N)

        # Sample from the last step
        return Y_N[-1]

    def sample_one_backbone_step(
        self,
        backbone: nn.Module,
        head: nn.Module,
        X_t: Tensor,
        t: Tensor,
        conditions: Tensor | None = None,
        solver: str = "euler",
        head_steps: int | None = None,
    ):
        h_t = backbone(X_t, t, conditions)
        Y = self.sample_head(head, h_t, t, steps=head_steps, solver=solver)

        # Update the X_t
        T_inv = 1 / self.backbone_disc_T
        X_t = X_t + Y * T_inv

        return X_t

    @torch.no_grad()
    def sample_loop(
        self,
        backbone: nn.Module,
        head: nn.Module,
        X_0: Tensor,
        conditions: Tensor | None = None,
        solver: str = "euler",
        ret_sample_process: bool = False,
    ):
        for t in range(self.backbone_disc_T):
            X_t = X_0
            bs = X_0.shape[0]
            t = torch.full((bs,), t, device=X_0.device)

            X_ts = []

            # Sample one step
            X_t = self.sample_one_backbone_step(backbone, head, X_t, t, conditions, solver=solver)

            if ret_sample_process:
                X_ts.append(X_t)

        if ret_sample_process:
            return X_t, torch.stack(X_ts)
        else:
            return X_t
