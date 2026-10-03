# Breast Cancer Prediction -

## Project Description

This project implements a Machine Learning classification model for predicting whether a breast tumor is Malignant or Benign.

The project uses the Breast Cancer Wisconsin dataset provided through Scikit-learn.

The target variable is:

- `0` → Malignant
- `1` → Benign

The dataset contains 569 records and 30 real-valued features. :contentReference[oaicite:0]{index=0}

## Dataset

The dataset is loaded using the `load_breast_cancer()` method from Scikit-learn, as specified in the assignment.

```python
from sklearn.datasets import load_breast_cancer
