import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .ml.model import compute_model_metrics, inference, train_model


@pytest.fixture
def get_model():

    X = np.array([[1, 1]])
    y = np.array([1])
    model = train_model(X, y)

    return model, X, y


def test_train_model(get_model):

    model, _, _ = get_model

    assert type(model) == RandomForestClassifier


def test_inference(get_model):

    model, X, y = get_model

    assert inference(model, X) == np.array(y)


@pytest.mark.parametrize(
    "args",
    [
        ([np.array([1, 1]), np.array([1, 1]), 1, 1, 1]),
        ([np.array([0, 0]), np.array([1, 1]), 0, 1, 0]),
    ],
)
def test_compute_model_metrics(args):

    y, preds, exp_pre, exp_re, exp_fb = args
    pre, re, fb = compute_model_metrics(y, preds)

    assert pre == exp_pre
    assert re == exp_re
    assert fb == exp_fb
