import warnings
import torch.nn as nn

conv_cfg = {
    'Conv': nn.Conv2d,
    # TODO
}

def build_conv_layer(cfg, *args, **kwargs):
    if cfg is None:
        cfg_ = dict(type='Conv')
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_ = cfg.copy()
    
    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]
    layer = conv_layer(*args, **kwargs, **cfg_)
    return layer

class ConvModule(nn.Module):
    
