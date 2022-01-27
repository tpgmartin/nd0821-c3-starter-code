import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def compute_model_metrics_by_slice(test, cat_features, y, preds):
    """
    Validates the trained machine learning model calculating precision, recall,
    and F1 by slice.

    Inputs
    ------
    test : pd.DataFrame
        Test dataset 
    cat_features : list
        List of categorical feature column names
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    model_metrics_by_slice : pd.DataFrame
    """
    cats = []
    slices = []
    slice_precision = []
    slice_recall = []
    slice_fbeta = []
    for cat in cat_features:

        for slice in test[cat].unique():

            precision, recall, fbeta = compute_model_metrics(y[test[cat] == slice],preds[test[cat] == slice])

            cats.append(cat)
            slices.append(slice)
            slice_precision.append(precision)
            slice_recall.append(recall)
            slice_fbeta.append(fbeta)

    model_metrics_by_slice = pd.DataFrame({
        'cat_feature': cats,
        'slice': slices,
        'precision': precision,
        'recall': recall,
        'fbeta': fbeta
    })

    return model_metrics_by_slice


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds
