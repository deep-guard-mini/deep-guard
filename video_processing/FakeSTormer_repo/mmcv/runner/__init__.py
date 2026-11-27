import torch
def _load_checkpoint(filename, map_location=None, logger=None): return torch.load(filename, map_location=map_location)
def auto_fp16(*a, **kw): return lambda fn: fn
