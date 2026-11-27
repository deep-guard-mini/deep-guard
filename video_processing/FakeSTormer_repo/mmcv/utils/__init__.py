import logging
def to_2tuple(x): return (x, x) if isinstance(x, int) else tuple(x)
def get_logger(name, log_file=None, log_level=logging.INFO, file_mode="w"):
    l = logging.getLogger(name)
    if not l.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        l.addHandler(h)
    l.setLevel(log_level)
    return l
