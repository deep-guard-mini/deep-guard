import math, torch.nn as nn
from timm.layers import trunc_normal_
def build_conv_layer(cfg, *a, **kw):
    c = (cfg or {}).copy(); c.pop('type', None); return nn.Conv2d(*a, **kw, **c)
def build_upsample_layer(cfg, *a, **kw):
    c = cfg.copy(); t = c.pop('type', 'deconv')
    return nn.ConvTranspose2d(*a, **kw, **c) if t == 'deconv' else nn.Upsample(*a, **kw, **c)
def build_norm_layer(cfg, num_features, postfix=''):
    t = cfg.get('type', 'BN')
    return f'norm{postfix}', (nn.LayerNorm(num_features) if t == 'LN' else nn.BatchNorm2d(num_features))
def build_activation_layer(cfg):
    t = (cfg or {}).get('type', 'ReLU')
    return nn.GELU() if t == 'GELU' else (nn.SiLU() if t == 'SiLU' else nn.ReLU(inplace=True))
def constant_init(m, val, bias=0):
    if getattr(m, 'weight', None) is not None: nn.init.constant_(m.weight, val)
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, bias)
def normal_init(m, mean=0, std=1, bias=0):
    if getattr(m, 'weight', None) is not None: nn.init.normal_(m.weight, mean, std)
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, bias)
def trunc_normal_init(m, mean=0., std=1., a=-2., b=2., bias=0.):
    if getattr(m, 'weight', None) is not None: trunc_normal_(m.weight, mean=mean, std=std, a=a, b=b)
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, bias)
def xavier_init(m, gain=1, bias=0, distribution='normal'):
    if getattr(m, 'weight', None) is not None:
        (nn.init.xavier_uniform_ if distribution == 'uniform' else nn.init.xavier_normal_)(m.weight, gain=gain)
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, bias)
def kaiming_init(m, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    if getattr(m, 'weight', None) is not None:
        (nn.init.kaiming_uniform_ if distribution == 'uniform' else nn.init.kaiming_normal_)(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, bias)
def bias_init_with_prob(p): return float(-math.log((1 - p) / p))
