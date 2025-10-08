from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safe_load_file


class TaskState:
    CKPT_IGNORE = ["accelerator", "cfg", "models", "optimizer"]
    single_instance = None

    def __new__(cls, **kwargs):
        if cls.single_instance is None:
            cls.single_instance = super().__new__(cls)
            cls.single_instance._init_vars()
        cls.single_instance.register_vars(**kwargs)
        return cls.single_instance

    def print(self, *args, **kwargs):
        if self.accelerator is not None:
            self.accelerator.print(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def _init_vars(self):
        # Ensure some attributes exist
        self.cfg = None  # pylint: disable=W0201
        self.accelerator = None  # pylint: disable=W0201
        self.logger = None  # pylint: disable=W0201
        self.models = {}  # pylint: disable=W0201
        self.registered_models = []

    def register_vars(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __deepcopy__(self, memo):
        return self  # Can't be copied, only reconstructed

    def _make_state_dict(self, obj):
        if isinstance(obj, Mapping):
            return {k: self._make_state_dict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_state_dict(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._make_state_dict(v) for v in obj)
        if hasattr(obj, "state_dict"):
            return obj.state_dict()
        if isinstance(obj, (int, float, str, type(None))):
            return obj
        raise ValueError(f"Cannot make state dict from object of type {type(obj)}")

    def _load_state_dict(self, obj, state_dict):
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                if k in state_dict:
                    obj[k] = self._load_state_dict(v, state_dict[k])
        elif isinstance(obj, list):
            assert len(obj) == len(state_dict)
            for i, v in enumerate(obj):
                obj[i] = self._load_state_dict(v, state_dict[i])
        elif isinstance(obj, tuple):
            assert len(obj) == len(state_dict)
            return type(obj)(
                [self._load_state_dict(v, state_dict[i]) for i, v in enumerate(obj)]
            )
        elif hasattr(obj, "load_state_dict"):
            obj.load_state_dict(state_dict)
        else:
            assert type(obj) is type(state_dict), (
                f"Cannot load state dict into object of different type {type(obj)} != {type(state_dict)}"
            )
        return obj

    def state_dict(self):
        # Won't save any field in CKPT_IGNORE or starting with '_'
        save_fields = {
            k: v
            for k, v in self.__dict__.items()
            if k not in self.CKPT_IGNORE and not k.startswith("_")
        }
        sd = self._make_state_dict(save_fields)
        return sd

    def load_state_dict(self, state_dict):
        self._load_state_dict(self.__dict__, state_dict)


# Initialization utils


def load_submodule(
    model,
    weights_path,
    module_name,
    strict=True,
    accelerator=None,
    map_fn=None,
    default_load="ae_ema",
):
    warn = accelerator.warning if accelerator else lambda x: print("[WARNING", x)
    assert isinstance(weights_path, (str, Path))
    if len(Path(weights_path).parts) == 1:
        ckpt_path = TaskState().cfg.ckpt_dir
        weights_path = (
            Path(ckpt_path)
            / "jobs"
            / str(weights_path)
            / "checkpoints"
            / "best"
            / f"model_{default_load}.safetensors"
        )

    weights = safe_load_file(weights_path)

    if module_name:
        if not module_name.endswith("."):
            module_name += "."
        weights = {
            k.replace(module_name, ""): v
            for k, v in weights.items()
            if k.startswith(module_name)
        }
        if map_fn is not None:
            weights, unmaped_weight = {}, weights
            for k, v in unmaped_weight.items():
                k, v = map_fn(k, v)
                if k:
                    weights[k] = v

    if not strict:
        cur_weights = model.state_dict()
        for k in list(weights.keys()):
            if k not in cur_weights:
                del weights[k]
                warn(f"{k} not found in model state_dict, skipping loading.")
            elif weights[k].shape != cur_weights[k].shape:
                warn(
                    f"Shape mismatch for {k}: {weights[k].shape} != {cur_weights[k].shape}, skipping loading."
                )
                del weights[k]

    model.load_state_dict(weights, strict=strict)
    mark_init_state(model, True)

    return len([v.numel() for v in weights.values()])


def mark_init_state(model, init_state):
    """Mark all modules of a model as initialized or not. Can be used to reinitialize the model, or protect weights of a submodule."""
    n_mod = 0
    for m in model.modules():
        if not hasattr(m, "_w_init") or m._w_init != init_state:
            n_mod += 1
        m._w_init = init_state
    return n_mod


@torch.no_grad()
def init_weights(
    model,
    method=None,
    nonlinearity="leaky_relu",
    init_embeds=False,
    scale=0.02,
    force=False,
    checkpoint=None,
    ckpt_module=None,
    ckpt_args=None,
    freeze=False,
):
    """
    Initialize weights of a model with a given method, when the model is not already initialized. Mark initialized modules with _w_init=True.
    method: str, default="auto"
        - "auto": use kaiming_uniform for conv and linear, and normal for embeddings
        - "kaiming_uniform", "kaiming_normal", "xavier_uniform", "xavier_normal", "uniform", "normal"
        - float value: normal with the given scale
    method modifiers:
        - "+nonlinearity-leaky_relu", "+nonlinearity-relu", "+nonlinearity-tanh", "+nonlinearity-sigmoid"
        - "+scale-0.02", "+scale-0.1", etc.
        - "+conv-kaiming_uniform", "+conv-kaiming_normal", "+conv-xavier_uniform", "+conv-xavier_normal", "+conv-uniform", etc.

    """

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # If checkpoint is provided, load weights from the checkpoint
    if checkpoint is not None:
        n_mod = load_submodule(
            model, checkpoint, ckpt_module or "", **(ckpt_args or {})
        )
        mark_init_state(model, True)
        return n_mod

    # Parse method
    method = method.lower() if isinstance(method, str) else method
    if method in ["default", "auto"]:
        # Default initialization, mark initialized modules
        n_mod = mark_init_state(model, True)  # Keep default weights
        return n_mod

    if method in [None, "none", "skip"]:
        # Don't initialize, module can be initialized by parent
        return

    if isinstance(method, (int, float)):
        scale = method
        method = "normal"

    # Express modifiers
    conv_method = "normal"  # default
    method_parts = method.split("+")
    for part in method_parts:
        if "-" in part:
            part, value = part.split("-", 1)
            if part == "scale":
                scale = float(value)
            elif part == "nonlinearity":
                nonlinearity = value
            elif part == "conv":
                conv_method = value
            else:
                raise ValueError(f"Unknown part {part} in method {method}")
        else:
            method = part

    # Initialize weights function
    def init_layer_weights(l_method, layer_weight):
        if l_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(layer_weight, nonlinearity=nonlinearity)
        elif l_method == "kaiming_normal":
            nn.init.kaiming_normal_(layer_weight, nonlinearity=nonlinearity)
        elif l_method == "xavier_uniform":
            nn.init.xavier_uniform_(layer_weight)
        elif l_method == "xavier_normal":
            nn.init.xavier_normal_(layer_weight)
        elif l_method == "uniform":
            nn.init.uniform_(layer_weight, a=-scale, b=scale)
        elif l_method == "normal":
            nn.init.normal_(layer_weight, mean=0.0, std=scale)
        elif l_method == "zero":
            nn.init.zeros_(layer_weight)
        else:
            raise ValueError(f"Unknown weights initialization method {l_method}")

    # Initialize weights
    n_init = 0
    for m in model.modules():
        if (hasattr(m, "_w_init") and m._w_init) or force:
            continue
        if isinstance(m, nn.Conv2d):
            init_layer_weights(conv_method or method, m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init_layer_weights(method, m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            if init_embeds:
                nn.init.normal_(m.weight, mean=0)
                m.weight.data = (
                    nn.functional.normalize(m.weight.data, p=2, dim=-1) * scale
                )
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                m.bias.data.zero_()
            if m.weight is not None:
                m.weight.data.fill_(1.0)
        elif "RMSNorm" in m.__class__.__name__:
            if hasattr(m, "weight") and m.weight is not None:
                m.weight.data.fill_(1.0)
        else:
            m._w_init = True
            continue
        m._w_init = True
        n_init += 1
    return n_init


@torch.no_grad()
def init_zero(model):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
            m.weight.data.zero_()
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            m.weight.data.zero_()
        elif "RMSNorm" in m.__class__.__name__:
            m.weight.data.zero_()
        elif hasattr(m, "lambda1"):
            nn.init.zeros_(m.lambda1.data)
        m._w_init = True
