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
        # --- 1. 在开头添加这几行，获取图像尺寸信息 ---
        if to_1d:
            _, _, H, W = X_T.shape
        # --------------------------------------------

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
        t = torch.randint(0, T, (bsz,), device=X_T.device)

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
        s = torch.rand(bsz * seq_len, device=Y.device)
        Y_s = (1 - s).view(-1, 1) * Y_0 + s.view(-1, 1) * Y

        # Head forward
        u = head(h_t, t, Y_s, s)
        loss = torch.nn.functional.mse_loss(u, Y - Y_0)

        # # To predicted Y
        # X_pred = u + Y_0 + X_0
        # if to_1d:
        #     X_pred = rearrange(
        #         X_pred,
        #         "b (h w) (dh dw c) -> b c (h dh) (w dw)",
        #         dh=patch_size,
        #         dw=patch_size,
        #     )

        # To predicted Y
        # 1. 首先，将 X_0 也展平成 2D，使其维度与 u 和 Y_0 匹配，然后相加
        X_pred_flat = u + Y_0 + X_0.view(bsz * seq_len, -1)

        # `to_1d` 在你的训练脚本中恒为 True，所以我们主要关心 if 内部
        if to_1d:
            # 2. 将相加后的结果从 2D 还原回 3D ([B, SeqLen, Dim])，以供 rearrange 使用
            X_pred = X_pred_flat.view(bsz, seq_len, -1)

            # 3. 现在可以安全地进行 rearrange 操作，将序列还原为图像
            X_pred = rearrange(
                X_pred,
                "b (h w) (dh dw c) -> b c (h dh) (w dw)",
                dh=patch_size,
                dw=patch_size,
                h=(H // patch_size),
                w=(W // patch_size),
            )
        else:
            # 如果 to_1d 为 False，也最好将形状还原
            X_pred = X_pred_flat.view(bsz, seq_len, -1)

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

        # --- 1. 从 head 模型中直接获取正确的图块维度 (patch_dim) ---
        # input_projector 的输入维度正是我们需要的 patch_dim (256)
        patch_dim = head.input_projector.in_features

        # Start from Gaussian
        # Y_0 = torch.randn_like(h_t)

        Y_0 = torch.randn(
            h_t.shape[0],  # 批次维度和 h_t 保持一致
            patch_dim,  # 使用正确的特征维度 (256)
            device=h_t.device,
            dtype=h_t.dtype,
        )

        # Prefix head's inputs
        # call_fn = lambda Y_s, s: head(h_t, t, Y_s, s)
        call_fn = lambda s, Y_s: head(h_t, t, Y_s, s)

        # Time grid
        Ns = torch.linspace(0, 1, steps + 1, device=h_t.device)
        Y_N = odeint(call_fn, Y_0, Ns, method=solver)
        Y_N = cast(Tensor, Y_N)

        # Sample from the last step
        return Y_N[-1]

    # def sample_one_backbone_step(
    #     self,
    #     backbone: nn.Module,
    #     head: nn.Module,
    #     X_t: Tensor,
    #     t: Tensor,
    #     conditions: Tensor | None = None,
    #     solver: str = "euler",
    #     head_steps: int | None = None,
    # ):
    #     h_t = backbone(X_t, t, conditions)
    #     Y = self.sample_head(head, h_t, t, steps=head_steps, solver=solver)
    #
    #     # Update the X_t
    #     T_inv = 1 / self.backbone_disc_T
    #     X_t = X_t + Y * T_inv
    #
    #     return X_t
    #
    # @torch.no_grad()
    # def sample_loop(
    #     self,
    #     backbone: nn.Module,
    #     head: nn.Module,
    #     X_0: Tensor,
    #     conditions: Tensor | None = None,
    #     solver: str = "euler",
    #     ret_sample_process: bool = False,
    # ):
    #     for t in range(self.backbone_disc_T):
    #         X_t = X_0
    #         bs = X_0.shape[0]
    #         t = torch.full((bs,), t, device=X_0.device)
    #
    #         X_ts = []
    #
    #         # Sample one step
    #         X_t = self.sample_one_backbone_step(
    #             backbone, head, X_t, t, conditions, solver=solver
    #         )
    #
    #         if ret_sample_process:
    #             X_ts.append(X_t)
    #
    #     if ret_sample_process:
    #         return X_t, torch.stack(X_ts)
    #     else:
    #         return X_t

    def sample_one_backbone_step(
        self,
        backbone: nn.Module,
        head: nn.Module,
        X_t: Tensor,  # Input is an image [B, C, H, W]
        t: Tensor,
        patch_size: int,  # <--- 1. 添加 patch_size 参数
        conditions: Tensor | None = None,
        solver: str = "euler",
        head_steps: int | None = None,
    ):
        # 获取图像和序列尺寸，用于后续还原
        bsz, _, H, W = X_t.shape
        seq_len = (H // patch_size) * (W // patch_size)

        # 2. 将输入图像 X_t 转换为序列，以送入 backbone
        X_t_seq = rearrange(
            X_t,
            "b c (h dh) (w dw) -> b (h w) (dh dw c)",
            dh=patch_size,
            dw=patch_size,
        )

        # 3. backbone 接收序列，输出序列 h_t
        h_t = backbone(X_t_seq, t, conditions)

        # 4. 为 head 准备展平的输入
        h_t_flat = h_t.view(bsz * seq_len, -1)
        t_flat = t.repeat_interleave(seq_len)
        Y_flat = self.sample_head(head, h_t_flat, t_flat, steps=head_steps, solver=solver)

        # 5. 将 head 输出的展平向量 Y_flat 还原成图像形状
        Y_img = Y_flat.view(bsz, seq_len, -1)
        Y_img = rearrange(
            Y_img,
            "b (h w) (dh dw c) -> b c (h dh) (w dw)",
            dh=patch_size,
            dw=patch_size,
            h=(H // patch_size),
            w=(W // patch_size),
        )

        # 6. 用还原后的 Y_img 更新图像 X_t
        T_inv = 1 / self.backbone_disc_T
        X_t_new = X_t + Y_img * T_inv

        return X_t_new

    @torch.no_grad()
    def sample_loop(
        self,
        backbone: nn.Module,
        head: nn.Module,
        X_0: Tensor,
        patch_size: int,  # <--- 添加 patch_size 参数
        conditions: Tensor | None = None,
        solver: str = "euler",
        ret_sample_process: bool = False,
    ):
        X_t = X_0  # X_t 在循环外初始化
        X_ts = []
        if ret_sample_process:
            X_ts.append(X_t.clone())  # 保存初始状态

        for t_step in range(self.backbone_disc_T):
            bs = X_t.shape[0]
            t = torch.full((bs,), t_step, device=X_t.device, dtype=torch.long)

            # Sample one step, 传入 patch_size
            X_t = self.sample_one_backbone_step(backbone, head, X_t, t, patch_size, conditions, solver=solver)

            if ret_sample_process:
                X_ts.append(X_t.clone())

        if ret_sample_process:
            return X_t, torch.stack(X_ts)
        else:
            return X_t
