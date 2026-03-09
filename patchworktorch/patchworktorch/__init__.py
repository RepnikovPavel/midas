from .pycode import patchworktorch_class

def ground_filter(pts, **kwargs):
    gf = patchworktorch_class(kwargs)
    return gf.forward(pts)

__all__ = ['patchworktorch_class', 'ground_filter']