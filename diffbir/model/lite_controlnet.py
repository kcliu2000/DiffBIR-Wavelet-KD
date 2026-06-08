import torch
from torch import nn


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class LiteBlock(nn.Module):
    """
    輕量 residual block：
    GroupNorm + depthwise conv + pointwise conv
    """
    def __init__(self, channels):
        super().__init__()
        groups = 32 if channels >= 32 else 1
        self.norm = nn.GroupNorm(groups, channels)
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.norm(x)
        h = self.act(h)
        h = self.dw(h)
        h = self.act(h)
        h = self.pw(h)
        return x + h


class StandardBlock(nn.Module):
    """
    較強 residual block：
    GroupNorm + 3x3 conv + 3x3 conv

    用在 1280 stage，讓 student 不會過度壓縮。
    """
    def __init__(self, channels):
        super().__init__()
        groups = 32 if channels >= 32 else 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return x + h


class LiteControlNet(nn.Module):
    """
    Student IRControlNet / LiteControlNet.

    輸出 13 個 control tensors，對齊 DiffBIR ControlledUnetModel。

    對 crop_size=256，latent size=32x32，輸出 shapes 為：

    0:  320  @ 32x32
    1:  320  @ 32x32
    2:  320  @ 32x32
    3:  320  @ 16x16
    4:  640  @ 16x16
    5:  640  @ 16x16
    6:  640  @ 8x8
    7:  1280 @ 8x8
    8:  1280 @ 8x8
    9:  1280 @ 4x4
    10: 1280 @ 4x4
    11: 1280 @ 4x4
    12: 1280 @ 4x4

    heavy_1280:
      False -> very small student
      True  -> medium student, around 50% control-module reduction
    """

    def __init__(
        self,
        image_size=32,
        in_channels=4,
        hint_channels=4,
        model_channels=320,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attention_resolutions=(4, 2, 1),
        num_head_channels=64,
        use_spatial_transformer=True,
        use_linear_in_transformer=True,
        transformer_depth=1,
        context_dim=1024,
        legacy=False,
        base_channels=320,
        heavy_1280=False,
        **kwargs,
    ):
        super().__init__()

        self.dtype = torch.float32
        self.heavy_1280 = heavy_1280

        c1 = base_channels          # 320
        c2 = base_channels * 2      # 640
        c3 = base_channels * 4      # 1280

        Block1280 = StandardBlock if heavy_1280 else LiteBlock

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels + hint_channels, c1, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, 3, padding=1),
        )

        # 32x32, 320
        self.b0 = LiteBlock(c1)
        self.z0 = zero_module(nn.Conv2d(c1, c1, 1))

        self.b1 = LiteBlock(c1)
        self.z1 = zero_module(nn.Conv2d(c1, c1, 1))

        self.b2 = LiteBlock(c1)
        self.z2 = zero_module(nn.Conv2d(c1, c1, 1))

        # 32 -> 16, 320
        self.down0 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)
        self.z3 = zero_module(nn.Conv2d(c1, c1, 1))

        # 16x16, 640
        self.proj1 = nn.Conv2d(c1, c2, 1)

        self.b4 = LiteBlock(c2)
        self.z4 = zero_module(nn.Conv2d(c2, c2, 1))

        self.b5 = LiteBlock(c2)
        self.z5 = zero_module(nn.Conv2d(c2, c2, 1))

        # 16 -> 8, 640
        self.down1 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)
        self.z6 = zero_module(nn.Conv2d(c2, c2, 1))

        # 8x8, 1280
        self.proj2 = nn.Conv2d(c2, c3, 1)

        self.b7 = Block1280(c3)
        self.z7 = zero_module(nn.Conv2d(c3, c3, 1))

        self.b8 = Block1280(c3)
        self.z8 = zero_module(nn.Conv2d(c3, c3, 1))

        # 8 -> 4, 1280
        self.down2 = nn.Conv2d(c3, c3, 3, stride=2, padding=1)
        self.z9 = zero_module(nn.Conv2d(c3, c3, 1))

        # 4x4, 1280
        self.b10 = Block1280(c3)
        self.z10 = zero_module(nn.Conv2d(c3, c3, 1))

        self.b11 = Block1280(c3)
        self.z11 = zero_module(nn.Conv2d(c3, c3, 1))

        self.middle_block = Block1280(c3)
        self.middle_zero = zero_module(nn.Conv2d(c3, c3, 1))

    def forward(self, x, hint, timesteps=None, context=None):
        x = x.to(self.dtype)
        hint = hint.to(self.dtype)

        h = torch.cat([x, hint], dim=1)
        h = self.stem(h)

        outs = []

        h = self.b0(h)
        outs.append(self.z0(h))

        h = self.b1(h)
        outs.append(self.z1(h))

        h = self.b2(h)
        outs.append(self.z2(h))

        h = self.down0(h)
        outs.append(self.z3(h))

        h = self.proj1(h)

        h = self.b4(h)
        outs.append(self.z4(h))

        h = self.b5(h)
        outs.append(self.z5(h))

        h = self.down1(h)
        outs.append(self.z6(h))

        h = self.proj2(h)

        h = self.b7(h)
        outs.append(self.z7(h))

        h = self.b8(h)
        outs.append(self.z8(h))

        h = self.down2(h)
        outs.append(self.z9(h))

        h = self.b10(h)
        outs.append(self.z10(h))

        h = self.b11(h)
        outs.append(self.z11(h))

        h = self.middle_block(h)
        outs.append(self.middle_zero(h))

        assert len(outs) == 13, f"LiteControlNet must return 13 controls, got {len(outs)}"
        return outs
