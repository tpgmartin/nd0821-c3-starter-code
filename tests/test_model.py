import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from starter.starter.ml.model import compute_model_metrics, inference


def test_train_model(trainedModel):

    model, _, _ = trainedModel

    assert type(model) == RandomForestClassifier


def test_inference(trainedModel):

    model, X, y = trainedModel

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
