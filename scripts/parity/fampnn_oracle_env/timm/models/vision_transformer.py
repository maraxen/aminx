"""Oracle-env stub: verbatim copy of timm 1.0.31 ``timm/layers/mlp.py::Mlp`` (use_conv=False path).

Upstream FAMPNN imports ``timm.models.vision_transformer.Mlp`` (fampnn/model/sidechain_diffusion/
dit_utils.py:8). Importing real timm pulls torchvision, whose ABI does not match the titanix torch.
This file is the only substitution; every forward in this class is timm's own code.
"""

from functools import partial

from torch import nn


def _to_2tuple(x):
  return tuple(x) if isinstance(x, (list, tuple)) else (x, x)


class Mlp(nn.Module):
  def __init__(
    self,
    in_features,
    hidden_features=None,
    out_features=None,
    act_layer=nn.GELU,
    norm_layer=None,
    bias=True,
    drop=0.0,
    use_conv=False,
    device=None,
    dtype=None,
  ):
    dd = {"device": device, "dtype": dtype}
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    bias = _to_2tuple(bias)
    drop_probs = _to_2tuple(drop)
    linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
    self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0], **dd)
    self.act = act_layer()
    self.drop1 = nn.Dropout(drop_probs[0])
    self.norm = norm_layer(hidden_features, **dd) if norm_layer is not None else nn.Identity()
    self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1], **dd)
    self.drop2 = nn.Dropout(drop_probs[1])

  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x)
    x = self.norm(x)
    x = self.fc2(x)
    x = self.drop2(x)
    return x
