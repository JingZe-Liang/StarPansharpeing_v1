import ast
import enum
from typing import Callable

import numpy as np
import torch as th
from loguru import logger
from tqdm import tqdm

from . import path
from .integrators import get_timesteps, ode, sde
from .utils import mean_flat


class ModelType(str, enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = "noise"  # the model predicts epsilon
    SCORE = "score"  # the model predicts \nabla \log p(x)
    VELOCITY = "velocity"  # the model predicts v(x)
    X1 = "x1"


class PathType(str, enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = "linear"
    GVP = "gvp"
    VP = "vp"


class WeightType(str, enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = "none"
    VELOCITY = "velocity"
    LIKELIHOOD = "likelihood"


class Transport:
    def __init__(
        self,
        *,
        model_type: str,
        path_type: str,
        loss_type: str,
        train_eps: float,
        sample_eps: float,
        time_sample_type: str = "uniform",
        cfm_factor=0.0,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.path_type = path_type
        self.time_sample_type = time_sample_type
        self.cfm_factor = cfm_factor

    def prior_logp(self, z):
        """
        Standard multivariate normal prior
        Assume z is batched
        """
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2.0 * np.log(2 * np.pi) - th.sum(x**2) / 2.0
        return th.vmap(_fn)(z)

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if type(self.path_sampler) in [path.VPCPlan]:
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) and (
            self.model_type != ModelType.VELOCITY or sde
        ):  # avoid numerical issue by taking a first semi-implicit step
            t0 = (
                eps
                if (diffusion_form == "SBDM" and sde)
                or self.model_type != ModelType.VELOCITY
                else 0
            )
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, x1, *, t_forced: th.Tensor | None = None):
        """Sampling x0 & t based on shape of x1 (if needed)
        Args:
          x1 - data point; [batch, *dim]
        """

        x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)

        if t_forced is not None:
            return t_forced.to(x1), x0, x1

        time_sample_type = self.time_sample_type
        if time_sample_type == "sigmoid":
            mean, std = 0.0, 1.0
            t = th.randn(x1.shape[0]) * std + mean
            t = th.sigmoid(t) * (t1 - t0) + t0
        elif time_sample_type == "uniform":
            t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        elif time_sample_type == "lognormal":
            # comes from EDM paper
            rnd_norm = th.randn((x1.shape[0],))
            sigma = rnd_norm.exp()
            if self.path_type == PathType.LINEAR:
                t = sigma / (1 + sigma)
            elif self.path_type == PathType.GVP:
                t = 2 / th.pi * th.atan(sigma)
            else:
                raise NotImplementedError()
            t = t * (t1 - t0) + t0
        elif isinstance(time_sample_type, str):
            # discrete timesteps, e.g., '[0,0.2,0.4,0.6,0.8,0.9999]'
            try:
                time_lst = ast.literal_eval(time_sample_type)
                self._discreate_time_sampled_lst = time_lst
                time_choices = th.tensor(time_lst)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse time_sample_type: {time_sample_type}"
                ) from e
            # ignore t0 and t1
            if th.min(time_choices) < 0 or th.max(time_choices) > 1:
                raise ValueError(
                    f"time_sample_type values must be in [0,1], but got {time_choices}"
                )
            indices = th.randint(0, len(time_choices), (x1.shape[0],))
            t = time_choices[indices]
        else:
            raise ValueError(f"Unknown time_sample_type: {time_sample_type}")
        t = t.to(x1)
        return t, x0, x1

    def training_losses(
        self,
        model,
        x1,
        model_kwargs=None,
        get_pred_x_clean: bool = True,
        *,
        t_forced: th.Tensor | None = None,
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """
        if model_kwargs is None:
            model_kwargs = {}

        t, x0, x1 = self.sample(x1, t_forced=t_forced)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        model_output = model(xt, t, **model_kwargs)

        if len(model_output.shape) == len(xt.shape) + 1:
            x0 = x0.unsqueeze(-1).expand(
                *([-1] * (len(x0.shape))), model_output.shape[-1]
            )
            xt = xt.unsqueeze(-1).expand(
                *([-1] * (len(xt.shape))), model_output.shape[-1]
            )
            ut = ut.unsqueeze(-1).expand(
                *([-1] * (len(ut.shape))), model_output.shape[-1]
            )
        B, C = xt.shape[:2]
        assert model_output.shape == (B, C, *xt.shape[2:]), (
            f"Expected model output shape to be (B, C, *xt.shape[2:]), "
            f"but got {model_output.shape} instead. "
            f"Batch size: {B}, Channels: {C}, Spatial dimensions: {xt.shape[2:]}"
        )

        terms = {}
        terms["pred"] = model_output
        if self.model_type == ModelType.VELOCITY:
            terms["loss"] = mean_flat(((model_output - ut) ** 2))
            if self.cfm_factor > 0:
                u_tilde = th.roll(ut, shifts=1, dims=0)
                terms["cfm_loss"] = self.cfm_factor * mean_flat(
                    ((model_output - u_tilde) ** 2)
                )
                terms["loss"] -= terms["cfm_loss"]

        elif self.model_type == ModelType.X1:
            terms["loss"] = th.nn.functional.l1_loss(model_output, x1)
        else:
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t**2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()

            if self.model_type == ModelType.NOISE:
                terms["loss"] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms["loss"] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))

        if get_pred_x_clean:
            alpha_t, d_alpha_t = self.path_sampler.compute_alpha_t(t)
            sigma_t, d_sigma_t = self.path_sampler.compute_sigma_t(t)
            if self.model_type == ModelType.NOISE:
                pred_v = self.path_sampler.get_velocity_from_noise(model_output, xt, t)
            elif self.model_type == ModelType.SCORE:
                pred_v = self.path_sampler.get_velocity_from_score(model_output, xt, t)
            elif self.model_type == ModelType.X1:
                pred_v = model_output - x0
            else:
                pred_v = model_output
            # velocity = d_alpha_t * x1 + d_sigma_t * x0
            # pred_x_clean = (velocity - d_sigma_t * x0) / d_alpha_t

            # pred_x_clean = (prev_v - d_sigma_t * x0) / d_alpha_t
            pred_x_clean = x0 + pred_v
            terms["pred_x_clean"] = pred_x_clean

        return terms

    def get_drift(self):
        """member function for obtaining the drift of the probability flow ODE"""

        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return -drift_mean + drift_var * model_output  # by change of variable

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return -drift_mean + drift_var * score

        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, (
                "Output shape from ODE solver must match input shape"
            )
            return model_output

        return body_fn

    def get_score(
        self,
    ):
        """member function for obtaining score of
        x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = (
                lambda x, t, model, **kwargs: model(x, t, **kwargs)
                / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
            )
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = (
                lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(
                    model(x, t, **kwargs), x, t
                )
            )
        else:
            raise NotImplementedError()

        return score_fn


class Sampler:
    """Sampler class for the transport model"""

    def __init__(
        self,
        transport: Transport,
        time_type="linear",
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """

        self.transport = transport
        if transport.model_type != ModelType.X1:
            self.drift = self.transport.get_drift()
            self.score = self.transport.get_score()
        else:
            logger.info(
                f"[Flow Matching]: model_type is {self.transport.model_type}, drift and score are not needed"
            )
        self.time_type = time_type

    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):
        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(
                x, t, form=diffusion_form, norm=diffusion_norm
            )
            return diffusion

        sde_drift = lambda x, t, model, **kwargs: self.drift(
            x, t, model, **kwargs
        ) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)

        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion

    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""

        if last_step is None:
            last_step_fn = lambda x, t, model, **model_kwargs: x
        elif last_step == "Mean":
            last_step_fn = (
                lambda x, t, model, **model_kwargs: x
                + sde_drift(x, t, model, **model_kwargs) * last_step_size
            )
        elif last_step == "Tweedie":
            alpha = (
                self.transport.path_sampler.compute_alpha_t
            )  # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = lambda x, t, model, **model_kwargs: x / alpha(t)[0][0] + (
                sigma(t)[0][0] ** 2
            ) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = (
                lambda x, t, model, **model_kwargs: x
                + self.drift(x, t, model, **model_kwargs) * last_step_size
            )
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
        temperature=1.0,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        - temperature: temperature scaling for the noise during sampling; default to 1.0
        """

        if self.transport.model_type == ModelType.X1:
            raise ValueError("Model type X1 is not supported for SDE sampling")

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
            temperature=temperature,
        )

        last_step_fn = self.__get_last_step(
            sde_drift, last_step=last_step, last_step_size=last_step_size
        )

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample

    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        sampling_time_type="uniform",
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        temperature=1.0,
        progress=False,
        clip_for_x1_pred: bool = True,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        - temperature: temperature scaling for the drift during sampling; default to 1.0
        """

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        if self.transport.model_type == ModelType.X1:
            assert not reverse, "Model type X1 reverse mode is not implemnted yet"

            # ignore the self.drift
            def _sample_fn_loop(x0, model_fn, **model_kwargs):
                # simple euler method
                xt = x0.clone()
                xt_s = []
                times = get_timesteps(t0, t1, num_steps, "uniform")
                logger.debug(f"[FM sampler]: Sampling times: {times}")

                for t in tqdm(times, desc="sampling ..."):
                    t = t.repeat(xt.size(0)).to(xt)
                    x1_pred = model_fn(xt, t, **model_kwargs)
                    delta_t = (t1 - t0) / num_steps
                    assert delta_t > 0, "t1 must be larger than t0"
                    if clip_for_x1_pred:
                        x1_pred.clip_(min=-1, max=1)
                    xt = xt + delta_t * (x1_pred - x0)
                    xt_s.append(xt)
                return th.stack(xt_s, dim=0)

            return _sample_fn_loop
        else:
            if reverse:
                drift = lambda x, t, model, **kwargs: self.drift(
                    x, th.ones_like(t) * (1 - t), model, **kwargs
                )
            else:
                drift = self.drift

            def drift_progress_wrapper_fn(drift_fn: Callable, tbar):
                def wrapper_fn(x, t, model, **model_kwargs):
                    drift = drift_fn(x, t, model, **model_kwargs)
                    tbar.update(1)
                    return drift

                return wrapper_fn

            if progress:
                tbar = tqdm(range(num_steps), desc="sampling")
                drift = drift_progress_wrapper_fn(drift, tbar)

            _ode = ode(
                drift=drift,
                t0=t0,
                t1=t1,
                sampler_type=sampling_method,
                num_steps=num_steps,
                atol=atol,
                rtol=rtol,
                temperature=temperature,
                time_type=sampling_time_type,
            )

            return _ode.sample

    def sample_ode_likelihood(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        temperature=1.0,
    ):
        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - temperature: temperature scaling for the drift during sampling; default to 1.0
        """

        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(
                    th.sum(self.drift(x, t, model, **model_kwargs) * eps), x
                )[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            temperature=temperature,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn
