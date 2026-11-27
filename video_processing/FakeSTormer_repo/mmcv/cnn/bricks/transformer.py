import torch.nn as nn
from timm.layers import DropPath
class FFN(nn.Module):
    def __init__(self, embed_dims=256, feedforward_channels=1024, act_cfg=None, ffn_drop=0., dropout_layer=None, **kw):
        super().__init__()
        self.fc1, self.act, self.drop, self.fc2 = nn.Linear(embed_dims, feedforward_channels), nn.GELU(), nn.Dropout(ffn_drop), nn.Linear(feedforward_channels, embed_dims)
    def forward(self, x, identity=None):
        out = self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))
        return out + (identity if identity is not None else x)
def build_dropout(cfg): return DropPath(cfg.get("drop_prob", 0.)) if cfg else nn.Identity()
