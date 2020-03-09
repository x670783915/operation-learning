import torch.nn as nn
import torch.utils.checkpoint as cp

from torch.nn.modules.batchnorm import _BatchNorm
from ..utils import build_conv_layer, build_norm_layer

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        style='pytorch',
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type='BN'),
        dcn=None,
        gcb=None,
        gen_attention=None
    ):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yer.'
        assert gen_attention is None, 'Not implemented yet.'
        assert gcb is None, 'Not implemented yet.'
        assert not with_cp

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            3,
            padding=1,
            bias=False
        )
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
    
    @property
    def norm1(self):
        return getattr(self, self.norm1_name)
    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out

## TODO
