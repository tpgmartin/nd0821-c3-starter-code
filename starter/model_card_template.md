# Model Card

This model card provides details of the model implementation, context of the data used, metrics, and other important details required to inform you of the appropriate use of the body of work. For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model used is a Random Forest Classifier as per the [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) implementation.

## Intended Use

The model is used for binary classification: predicting whether someone's salary is above or less than or equal to $50K. This model was created for parital completion of Udacity's Machine Learning DevOps Engineer Nanodegree Program to demonstrate how a model may be trained and deployed as a service. It is intended purely for educational purposes and not for production use.

## Data

The data used for the model is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/census+income), which provides over 48K records consisting of 14 attributes and a target variable, salary. A test train split was performed in the propotions of 0.8 and 0.2 respectively, using Scikit-Learn.

## Metrics

The evalution metrics for the model are as follows,

| Metric | Value (2 d.p.) |
| ------ | ----- |
| Precision | 0.71 |
| Recall    | 0.61 |
| Fbeta     | 0.66 |

## Ethical Considerations

The model is intended for educational purposes only, and not meant to be used to inform wider societal matters.

## Caveats and Recommendations

The model has not been specifically optimised in any way. Further investigation may consider why recall may be lower than precision and whether this is can be improved.
