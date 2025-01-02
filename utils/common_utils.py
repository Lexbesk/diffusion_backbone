import argparse


def str_none(value):
    if value.lower() in ['none', 'null', 'nil'] or len(value) == 0:
        return None
    else:
        return value


def str2bool(value):
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def round_floats(o):
    if isinstance(o, float): return round(o, 2)
    if isinstance(o, dict): return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [round_floats(x) for x in o]
    return o


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
