import math

import torch as th
from loguru import logger
from torchdiffeq import odeint


def _time_shift(
    img_seq_len: int,
    basic_timesteps: th.Tensor,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    x1=256,
    x2=4096,
):
    mean = (max_shift - base_shift) / (x2 - x1)
    bias = base_shift - mean * x1
    mu = mean * img_seq_len + bias

    # time shift
    sigma = 1.0
    shifted_timesteps = math.exp(mu) / (math.exp(mu) + (1 / basic_timesteps - 1) ** sigma)
    return shifted_timesteps


def get_timesteps(
    t0=0,
    t1=1,
    num_steps=100,
    time_type: str = "uniform",
    discrete_timesteps: th.Tensor | None = None,
):
    logger.debug(f"[FM sampler]: use timestep type: {time_type} with {num_steps} steps.")

    if discrete_timesteps is not None:
        # ensure the discreate timesteps are sorted
        t = discrete_timesteps.sort().values
        if t[0].item() != 0.0:
            t = th.cat([th.tensor([0.0]).to(t.device), t], dim=0)
        if t[-1].item() != 1.0:
            t = th.cat([t, th.tensor([1.0]).to(t.device)], dim=0)
        return t

    if time_type in ("uniform", "linear"):
        t = th.linspace(t0, t1, num_steps)
    else:
        t_typ = time_type.split("_")
        if isinstance(t_typ, list):
            kwargs = t_typ[1:]
            t_typ = t_typ[0]
        else:
            kwargs = None

        if t_typ == "pow":
            p = float(kwargs[0])
            assert p > 1.0, "Only power > 1.0 is supported"
            t = th.linspace(t1, t0, num_steps).pow(p)
            t = t.flip(-1)
        elif t_typ == "shift":
            # Takes from flux dev.1 code
            img_seq_len: int = int(kwargs[0])
            if len(kwargs) == 3:
                basic_shift, max_shift = float(kwargs[1]), float(kwargs[2])
            else:
                basic_shift, max_shift = 0.5, 1.15
            basic_ts = th.linspace(t1, t0, num_steps)
            t = _time_shift(img_seq_len, basic_ts, basic_shift, max_shift)
            t = t.flip(-1)
        elif t_typ == "linear_scaled":
            t = th.linspace(t1**0.5, t0**0.5, steps=num_steps) ** 2
            t = t.flip(-1)
        else:
            raise NotImplementedError(f"Time type {time_type} is not supported.")

    # Ensure the last timestep is 1.0 (with tolerance for floating point precision)
    if abs(float(t[-1]) - 1.0) > 1e-6:
        logger.warning(f"Timesteps are: {t}, the last timestep is {float(t[-1])}, set to 1.0.")
        t = th.cat([t, th.tensor([1.0]).to(t.device)], dim=0)
    else:
        # Ensure exact 1.0 for the last timestep
        t[-1] = 1.0

    return t


class sde:
    """SDE solver class"""

    def __init__(
        self,
        drift,
        diffusion,
        *,
        t0,
        t1,
        time_type="uniform",
        num_steps,
        sampler_type,
        temperature=1.0,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = get_timesteps(t0, t1, num_steps, time_type)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type
        self.temperature = temperature

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t
        dw = w_cur * th.sqrt(self.dt)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        x = mean_x + th.sqrt(2 * diffusion) * dw * self.temperature
        return x, mean_x

    def __Heun_step(self, x, _, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(self.dt) * self.temperature
        t_cur = th.ones(x.size(0)).to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return xhat + 0.5 * self.dt * (K1 + K2), xhat  # at last time point we do not perform the heun step

    def __forward_fn(self):
        """TODO: generalize here by adding all private functions ending with steps to it"""
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Smapler type not implemented.")

        return sampler

    def sample(self, init, model, **model_kwargs):
        """forward loop of sde"""
        x = init
        mean_x = init
        samples = []
        sampler = self.__forward_fn()
        for i, ti in enumerate(self.t[:-1]):
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)

                # Check for numerical instability
                if th.isnan(x).any() or th.isinf(x).any():
                    logger.error(
                        f"[SDE Sampling] Numerical instability detected at step {i}/{len(self.t) - 1}, t={ti:.4f}\\n"
                        f"  - x has NaN: {th.isnan(x).any()}, Inf: {th.isinf(x).any()}\n"
                        f"  - x range: [{x.min():.4f}, {x.max():.4f}]\n"
                        f"  - mean_x range: [{mean_x.min():.4f}, {mean_x.max():.4f}]\n"
                        f"  - dt: {self.dt:.6f}, temperature: {self.temperature}"
                    )
                    raise ValueError(
                        f"SDE sampling diverged at step {i}. "
                        "Try reducing temperature, increasing num_steps, or checking model outputs."
                    )

                samples.append(x)

        return samples


class ode:
    """ODE solver class"""

    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        time_type: str = "uniform",
        temperature=1.0,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift

        # Get sample times
        self.t = get_timesteps(t0, t1, num_steps, time_type)

        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type
        self.temperature = temperature

    def sample(self, x, model, **model_kwargs):
        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn(t, x):
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
            # For ODE, we scale the drift by the temperature
            # This is equivalent to scaling time by 1/temperature
            model_output = self.drift(x, t, model, **model_kwargs)
            if self.temperature != 1.0:
                # If it's a tuple (for likelihood calculation), only scale the first element
                if isinstance(model_output, tuple):
                    scaled_output = (
                        model_output[0] / self.temperature,
                        model_output[1],
                    )
                    return scaled_output
                else:
                    return model_output / self.temperature
            return model_output

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(_fn, x, t, method=self.sampler_type, atol=atol, rtol=rtol)
        return samples
