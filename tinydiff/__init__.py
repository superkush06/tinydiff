"""tinydiff — reverse-mode autodiff in pure Python + NumPy."""

from . import nn, ops, optim
from .gradcheck import grad_check
from .loss import mse_loss, softmax_crossentropy
from .nn import Linear, Module, Sequential
from .ops import (
    add,
    div,
    exp,
    log,
    matmul,
    mean,
    mul,
    neg,
    pow_,
    relu,
    sigmoid,
    sub,
    sum_,
    tanh,
)
from .optim import SGD, Adam
from .tensor import Tensor

__version__ = "0.4.1"
__all__ = [
    "Tensor",
    "add", "sub", "mul", "div", "matmul", "pow_", "neg",
    "exp", "log", "relu", "sigmoid", "tanh", "sum_", "mean",
    "Linear", "Sequential", "Module",
    "mse_loss", "softmax_crossentropy",
    "SGD", "Adam",
    "grad_check",
    "nn", "ops", "optim",
    "__version__",
]
