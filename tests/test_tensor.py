"""Tensor + autograd tests."""


import tinydiff as td


def test_tensor_construct():
    t = td.Tensor([1.0, 2.0, 3.0])
    assert t.shape == (3,)
    assert t.grad is None
    assert not t.requires_grad


def test_requires_grad_propagates():
    a = td.Tensor(2.0, requires_grad=True)
    b = td.Tensor(3.0)
    c = a + b
    assert c.requires_grad


def test_add_backward():
    a = td.Tensor(2.0, requires_grad=True)
    b = td.Tensor(3.0, requires_grad=True)
    c = a + b
    c.backward()
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_mul_backward():
    a = td.Tensor(4.0, requires_grad=True)
    b = td.Tensor(5.0, requires_grad=True)
    c = a * b
    c.backward()
    assert a.grad == 5.0
    assert b.grad == 4.0


def test_chain_rule():
    a = td.Tensor(2.0, requires_grad=True)
    b = td.Tensor(3.0, requires_grad=True)
    c = (a * b + a)
    c.backward()
    # dc/da = b + 1 = 4 ; dc/db = a = 2
    assert a.grad == 4.0
    assert b.grad == 2.0
