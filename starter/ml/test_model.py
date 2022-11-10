"""
Unit test for trained model
"""
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference

import os
import pytest
import pandas as pd

def test_data():
    """
    DataFrame from census.csv file after
    """
    print(os.getcwd())
    dataframe = pd.read_csv("data/census.csv")
    assert True

@pytest.fixture
def data():
    """
    DataFrame from census.csv file after
    """
    dataframe = pd.read_csv("data/census.csv")

    return dataframe

@pytest.fixture
def train_test(data):
    """
    Train and test data set from dataFrame
    """
    train, test = train_test_split(data, test_size=0.20, random_state=23)

    return train, test

@pytest.fixture
def train_test_data(train_test):
    '''
    fixture for segregated dataset.
    '''

    train, test = train_test

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

    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features,
        label="salary", training=False, encoder=encoder, lb=lb
    )

    return X_train, y_train, X_test, y_test

@pytest.fixture
def model(train_test_data):
    '''
    fixture for a trained model
    '''

    X_train, y_train, _, _ = train_test_data
    model = train_model(X_train, y_train)
    return model

def test_train_model(model):
    """
    test for train_model()
    """
    try:
        assert hasattr(model, 'fit')
    except AssertionError as err:
        raise err

@pytest.fixture
def preds(train_test_data, model):
    '''
    Prediction for inference test
    '''
    _, _, X_test, _ = train_test_data

    # Inference
    preds = inference(model, X_test)

    return preds

def test_inference(train_test_data, preds):
    '''
    Inference test
    '''
    _, _, _, y_test = train_test_data

    try:
        assert y_test.size == preds.size
    except AssertionError as err:
        raise err
