"""Loss tests."""

import numpy as np

import tinydiff as td


def test_mse_zero_when_equal():
    pred = td.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    target = td.Tensor(np.array([1.0, 2.0, 3.0]))
    loss = td.mse_loss(pred, target)
    assert abs(loss.data - 0.0) < 1e-12


def test_mse_grad_matches_2_diff_over_n():
    pred = td.Tensor(np.array([2.0, 4.0]), requires_grad=True)
    target = td.Tensor(np.array([1.0, 1.0]))
    loss = td.mse_loss(pred, target)
    loss.backward()
    # grad = 2 * (pred - target) / N
    np.testing.assert_allclose(pred.grad, np.array([1.0, 3.0]))


def test_softmax_xent_correct_on_one_hot():
    """Loss should equal -log(softmax)[correct] averaged over batch."""
    logits = td.Tensor(np.array([[2.0, 1.0, 0.1]]), requires_grad=True)
    loss = td.softmax_crossentropy(logits, np.array([0]))
    softmax = np.exp([2.0, 1.0, 0.1]) / np.exp([2.0, 1.0, 0.1]).sum()
    assert abs(loss.data - (-np.log(softmax[0]))) < 1e-9


def test_softmax_xent_grad_sums_to_zero():
    logits = td.Tensor(np.array([[2.0, 1.0, 0.1], [0.0, 0.5, 0.5]]),
                       requires_grad=True)
    loss = td.softmax_crossentropy(logits, np.array([0, 1]))
    loss.backward()
    # Each row should sum to 0 (because grad of a probability simplex)
    np.testing.assert_allclose(logits.grad.sum(axis=1), 0.0, atol=1e-9)
