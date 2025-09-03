import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.helpers import to_2tuple

from .norm_act import SwiGLUAct


class SwiGLU(nn.Module):
    """SwiGLU
    NOTE: GluMLP above can implement SwiGLU, but this impl has split fc1 and
    better matches some other common impl which makes mapping checkpoints simpler.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer: type[nn.Module] = nn.SiLU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias: tuple[bool, bool] = to_2tuple(bias)  # type: ignore
        drop_probs: tuple[float, float] = to_2tuple(drop)  # type: ignore
        self.use_conv = use_conv

        linear_layer = (
            functools.partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        )
        self.fc1_g = linear_layer(in_features, hidden_features, bias=bias[0])
        self.fc1_x = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        if self.fc1_g.bias is not None:
            nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ClipSwiGLUMlp(SwiGLU):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer: type[nn.Module] = SwiGLUAct,
        norm_layer=None,
        bias=True,
        drop=0.0,
        mlp_bias=True,
        use_conv=False,
    ):
        super().__init__(
            in_features,
            hidden_features,
            out_features,
            act_layer,
            norm_layer,
            bias,
            drop,
            use_conv,
        )
        self.mlp_bias = (
            nn.Parameter(torch.zeros(self.fc2.weight.shape[0])) if mlp_bias else None
        )

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        if isinstance(self.act, SwiGLUAct):
            x = self.act(x_gate, x)
        else:
            x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        if self.mlp_bias is not None:
            if not self.use_conv:
                bias = self.mlp_bias
            else:
                bias = self.mlp_bias[..., None, None]
            x += bias
        x = self.drop2(x)
        return x
