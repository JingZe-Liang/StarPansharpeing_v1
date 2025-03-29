import accelerate
import torch

from src.stage1.discretization.collections.bsq import BinarySphericalQuantizer
from src.stage1.utilities.losses.gan_loss.loss import VQLPIPSWithDiscriminator

if __name__ == "__main__":
    # init accelerate
    accelerator = accelerate.Accelerator(
        mixed_precision="bf16",
    )

    bsq = BinarySphericalQuantizer(
        embed_dim=16,
        beta=0.0,
        gamma=1.0,
        gamma0=1.0,
        zeta=1.0,
        input_format="bchw",
        soft_entropy=True,
        group_size=8,
        persample_entropy_compute="group",
        cb_entropy_compute="group",
        l2_norm=True,
        inv_temperature=1,
    ).cuda()
    vq_loss = VQLPIPSWithDiscriminator(
        # discriminator
        disc_start=0,
        disc_factor=1.0,
        disc_weight=1.0,
        disc_reg_freq=1,
        disc_reg_r1=10,
        # disc network cfg
        disc_network_type="stylegan",
        disc_input_size=256,
        disc_in_channels=8,
        disc_num_layers=3,
        use_actnorm=False,
        disc_conditional=False,
        disc_ndf=64,
        disc_loss="hinge",
        # codebook losses
        quantizer_options={"quantizer_loss_weight": 0.1},
        # perceptual loss
        perceptual_weight=0.0,
        perceptual_type=None,
        # generator loss
        reconstruction_weight=1.0,
        gen_loss_weight=1.0,
        # quantizer losses
        quantizer_type="bsq",  # [bsq, vq]
        # other losses
        lecam_loss_weight=0.1,
        loss_ssim=False,
        ssim_weight=0.1,
        # if is video
        num_frames=1,
        # not reconstruction loss if using diffusion slots
        force_not_use_recon_loss=False,
    ).cuda()

    dtype = torch.bfloat16
    bs = 2
    z_from_enc = torch.randn(bs, 16, 32, 32).cuda().to(dtype)
    inputs = torch.randn(bs, 8, 256, 256).cuda().to(dtype)
    recon_from_dec = torch.randn(bs, 8, 256, 256).cuda().to(dtype)

    with torch.autocast("cuda", dtype=dtype):
        z_q, q_total_loss, q_info = bsq.forward(z_from_enc)
        print(z_q.shape)

        # to vq loss
        loss, loss_terms = vq_loss.forward(
            inputs,
            recon_from_dec,
            optimizer_idx=0,
            global_step=1,
            q_loss_total=q_total_loss,
            q_loss_breakdown=q_info,
            last_layer=None,
            cond=None,
            split="train",
        )

        # to vq loss
        loss, loss_terms = vq_loss.forward(
            inputs,
            recon_from_dec,
            optimizer_idx=1,
            global_step=1,
            q_loss_total=q_total_loss,
            q_loss_breakdown=q_info,
            last_layer=None,
            cond=None,
            split="train",
        )
