# Script to train machine learning model.

# Add the necessary imports for the starter code.
from ml.data import process_data
from .ml.model import compute_model_metrics, compute_model_metrics_by_slice, inference, train_model
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# Add code to load in the data.
dirname = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(dirname, '../data/census.csv'))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Save lb
pickle.dump(lb, os.path.join(dirname, '../model/lb.pkl'))

# Save encoder
pickle.dump(encoder, os.path.join(dirname, '../model/encoder.pkl'))

# Train and save a model.
model = train_model(X_train, y_train)
pickle.dump(model, os.path.join(dirname, '../model/model.pkl'))

preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(precision, recall, fbeta)
model_metrics_by_slice = compute_model_metrics_by_slice(
    test, cat_features, y_test, preds)

# save data
model_metrics_by_slice.to_csv(os.path.join(dirname, '../model/model_metrics.csv'), index=False)
