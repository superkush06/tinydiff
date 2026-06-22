"""nn.Module tests."""

import numpy as np

import tinydiff as td


def test_linear_forward_shape():
    layer = td.Linear(3, 5)
    x = td.Tensor(np.random.randn(8, 3))
    y = layer(x)
    assert y.shape == (8, 5)


def test_linear_params_collected():
    layer = td.Linear(3, 5)
    params = layer.parameters()
    sorted(id(p) for p in params)
    assert len(params) == 2  # W and b


def test_sequential_forward():
    model = td.Sequential(
        td.Linear(2, 4),
        td.relu,
        td.Linear(4, 1),
    )
    x = td.Tensor(np.random.randn(6, 2))
    y = model(x)
    assert y.shape == (6, 1)


def test_sequential_params_collected():
    model = td.Sequential(td.Linear(3, 4), td.Linear(4, 2))
    assert len(model.parameters()) == 4  # 2 layers × (W, b)
