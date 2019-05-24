import torch
from torch.autograd import Variable
import math
import numpy as np

class RangeNormalize(object):
    """
    Given min_val: (R, G, B) and max_val: (R,G,B),
    will normalize each channel of the th.*Tensor to
    the provided min and max values.
    Works by calculating :
        a = (max'-min')/(max-min)
        b = max' - a * max
        new_value = a * value + b
    where min' & max' are given values,
    and min & max are observed min/max for each channel

    Arguments
    ---------
    min_range : float or integer
        Min value to which tensors will be normalized
    max_range : float or integer
        Max value to which tensors will be normalized
    fixed_min : float or integer
        Give this value if every sample has the same min (max) and
        you know for sure what it is. For instance, if you
        have an image then you know the min value will be 0 and the
        max value will be 255. Otherwise, the min/max value will be
        calculated for each individual sample and this will decrease
        speed. Dont use this if each sample has a different min/max.
    fixed_max :float or integer
        See above
    Example:
        >>> x = th.rand(3,5,5)
        >>> rn = RangeNormalize((0,0,10),(1,1,11))
        >>> x_norm = rn(x)
    Also works with just one value for min/max:
        >>> x = th.rand(3,5,5)
        >>> rn = RangeNormalize(0,1)
        >>> x_norm = rn(x)
    """
    def __init__(self,
                 min_val,
                 max_val):
        """
        Normalize a tensor between a min and max value
        Arguments
        ---------
        min_val : float
            lower bound of normalized tensor
        max_val : float
            upper bound of normalized tensor
        """
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _min_val = _input.min()
            _max_val = _input.max()
            a = (self.max_val - self.min_val) / (_max_val - _min_val)
            b = self.max_val- a * _max_val
            _input = _input.mul(a).add(b)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]                                                 
