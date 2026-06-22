"""tinydiff — reverse-mode autodiff in pure Python + NumPy."""

from .gradcheck import grad_check
from .loss import mse_loss, softmax_crossentropy
from .nn import Linear, Module, Sequential
from .ops import add, div, exp, log, matmul, mean, mul, neg, pow_, relu, sigmoid, sub, sum_, tanh
from .optim import SGD, Adam
from .tensor import Tensor

__version__ = "0.1.0"
__all__ = ["Tensor", "add", "div", "exp", "log", "matmul", "mean", "mul", "neg", "pow_", "relu", "sigmoid", "sub", "sum_", "tanh", "Linear", "Module", "Sequential", "mse_loss", "softmax_crossentropy", "SGD", "Adam", "grad_check", "__version__"]
