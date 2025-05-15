import ast

import torch
from accelerate.state import AcceleratorState, is_initialized

from src.stage1.one_d_tokenizer.FlowMo.flowmo.models import (
    FlowMo,
    _edm_to_flow_convention,
)


def rf_sample(
    model,
    z,
    code,
    null_code=None,
    sample_steps=25,
    cfg=2.0,
    schedule="linear",
    cfg_interval: tuple | str = (0.17, 1.02),
):
    from tqdm import tqdm

    b = z.size(0)
    if schedule == "linear":
        ts = torch.arange(1, sample_steps + 1).flip(0) / sample_steps
        dts = torch.ones_like(ts) * (1.0 / sample_steps)
    elif schedule.startswith("pow"):
        p = float(schedule.split("_")[1])
        ts = torch.arange(0, sample_steps + 1).flip(0) ** (1 / p) / sample_steps ** (
            1 / p
        )
        dts = ts[:-1] - ts[1:]
    else:
        raise NotImplementedError

    if cfg_interval is None:
        interval = None
    else:
        if isinstance(cfg_interval, str):
            cfg_lo, cfg_hi = ast.literal_eval(cfg_interval)
        else:
            assert isinstance(cfg_interval, tuple)
            cfg_lo, cfg_hi = cfg_interval
            assert cfg_hi > cfg_lo, "cfg_hi should be greater than cfg_lo"
        interval = _edm_to_flow_convention(cfg_lo), _edm_to_flow_convention(cfg_hi)

    images = []
    for i, (t, dt) in tqdm(enumerate((zip(ts, dts))), leave=False, total=sample_steps):
        timesteps = torch.tensor([t] * b).to(z.device)
        vc, decode_aux = model.decode(img=z, timesteps=timesteps, code=code)

        if null_code is not None and (
            interval is None
            or ((t.item() >= interval[0]) and (t.item() <= interval[1]))
        ):
            vu, _ = model.decode(img=z, timesteps=timesteps, code=null_code)
            vc = vu + cfg * (vc - vu)

        z = z - dt * vc
        images.append(z)
    return images


class FlowMoTokenizer(FlowMo):
    def __init__(self, width, config):
        super().__init__(width, config)

        self.register_buffer("zero", torch.zeros(1))

    def forward_with_loss(
        self, x, inferece_with_n_slots=-1, aux_state: dict = None, **kwargs_discard
    ):
        config = self.config
        b = x.size(0)

        if config.opt.schedule == "lognormal":
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        elif config.opt.schedule == "fat_lognormal":
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
            t = torch.where(torch.rand_like(t) <= 0.9, t, torch.rand_like(t))
        elif config.opt.schedule == "uniform":
            t = torch.rand((b,), device=x.device)
        elif config.opt.schedule.startswith("debug"):
            p = float(config.opt.schedule.split("_")[1])
            t = torch.ones((b,), device=x.device) * p
        else:
            raise NotImplementedError

        t = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - t) * x + t * z1

        zt, t = zt.to(x.dtype), t.to(x.dtype)

        # posttrain_sample, original_code, q_loss
        # TODO: pred_x_clean, quan_info, diff_loss
        vtheta, aux = super().forward(
            img=x,
            noised_img=zt,
            timesteps=t.reshape((b,)),
        )

        diff = z1 - vtheta - x
        x_pred = zt - vtheta * t
        aux["pred_x_clean"] = x_pred

        loss = ((diff) ** 2).mean(dim=list(range(1, len(x.shape))))
        loss = loss.mean()

        # aux["loss_dict"] = {}
        # aux["loss_dict"]["diffusion_loss"] = loss
        # aux["loss_dict"]["quantizer_loss"] = aux["quantizer_loss"]

        # * lpips loss
        losses = {}
        losses["repa_loss"] = self.zero
        losses["diff_loss"] = loss
        losses["quantizer_loss"] = aux.pop("quantizer_loss", self.zero)

        # if config.opt.lpips_weight != 0.0:
        #     aux_loss = 0.0
        #     if config.model.posttrain_sample:
        #         x_pred = aux["posttrain_sample"]

        #     assert aux_state is not None, "aux_state is None"
        #     lpips_dist = aux_state["lpips_model"](x, x_pred)
        #     # lpips_dist = (config.opt.lpips_weight * lpips_dist).mean() + aux_loss
        #     lpips_dist = lpips_dist.mean() + aux_loss
        #     losses["lpips_loss"] = lpips_dist
        # else:
        #     lpips_dist = 0.0

        # loss = loss + aux["quantizer_loss"] + lpips_dist
        # aux["total_loss"] = loss

        return losses, aux

    @torch.no_grad()
    def forward_sample(self, images, dtype=None, code=None, **kwargs_discard):
        model = self
        config = self.config.eval.sampling
        dtype = (
            {
                "no": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }[AcceleratorState().mixed_precision]
            if is_initialized()
            else torch.bfloat16
        )

        with torch.autocast("cuda", dtype=dtype):
            bs, c, h, w = images.shape
            if code is None:
                x = images
                prequantized_code = model.encode(x)[0]
                code, *_ = model._quantize(prequantized_code)

            z = torch.randn((bs, self.image_channels, h, w)).cuda()

            mask = torch.ones_like(code[..., :1])
            code = torch.concatenate([code * mask, mask], axis=-1)

            cfg_mask = 0.0
            null_code = (
                code * cfg_mask if config.cfg != 1.0 else None
            )  # zero not learnable

            samples = rf_sample(
                model,
                z,
                code,
                null_code=null_code,
                sample_steps=config.sample_steps,
                cfg=config.cfg,
                schedule=config.schedule,
                cfg_interval=config.cfg_interval,
            )[-1].clip(-1, 1)

        _place_holder = {}
        return samples.to(torch.float32), _place_holder

    def forward(self, x, sample=False, **kwargs):
        if sample:
            return self.forward_sample(x, **kwargs)
        else:
            return self.forward_with_loss(x, **kwargs)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg_dict = {
        "data": {"image_size": 256},
        "model": {
            "code_length": 256,
            "enc_depth": 8,
            "dec_depth": 8,
            "patch_size": 8,
            "context_dim": 18,  # code dim
            "mup_width": 4,
            "enable_cfg": True,
            "posttrain_sample": False,
            "posttrain_sample_enable_cfg": 1.5,
            "posttrain_sample_k": 8,
            "enable_mup": False,
            "quantization_type": "noop",
            "enc_mup_width": None,
            "decoder_act_checkpoint": True,
            "encoder_act_checkpoint": True,
            "posttrain_sample": True,
        },
        "opt": {
            "schedule": "lognormal",
            "lpips_weight": 0,
        },
        "eval": {
            "reconstruction": True,
            "state_dict_key": "model_ema_state_dict",
            "eval_dir": "",
            "eval_baseline": "",
            "continuous": True,
            "force_ckpt_path": None,
            "subsample_rate": 1,
            "sampling": {
                "sample_steps": 25,
                "schedule": "pow_0.25",
                "cfg": 1.5,
                "mode": "rf",
                "cfg_interval": "(.17,1.02)",
            },
        },
    }

    cfg = OmegaConf.create(cfg_dict)
    print(cfg)
    dtype = torch.bfloat16

    torch.cuda.set_device(2)
    model = FlowMoTokenizer(4, cfg).cuda().to(dtype)
    x = torch.randn(8, 3, 512, 512).cuda().to(dtype)

    # * forward loss
    # with torch.autocast("cuda", dtype=dtype):
    #     losses, aux = model.forward_with_loss(x)
    # print(losses.keys())
    # print(aux.keys())

    # * forward sample
    with torch.autocast("cuda", dtype=dtype):
        samples = model.forward_sample(x)
    print(samples.shape)
    print(samples.dtype)
