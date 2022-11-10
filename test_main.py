from main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greeting": "Hello World!"}

def test_inference_small_salary():
    test_data = \
    {
        "age" : 39,
        "workclass" : "State-gov",
        "fnlgt" : 77516,
        "education" : "Bachelors",
        "education-num" : 13,
        "marital-status" : "Never-married",
        "occupation" : "Adm-clerical",
        "relationship" : "Not-in-family",
        "race" : "White",
        "sex" : "Male",
        "capital-gain" : 2174,
        "capital-loss" : 0,
        "hours-per-week" : 40,
        "native-country" : "United-States"
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200, \
        f"Response code is not successful. {response.json()}"
    assert response.json() == {"prediction":"<=50K"}

def test_inference_large_salary():
    test_data = \
    {
        "age" : 30,
        "workclass" : "State-gov",
        "fnlgt" : 141297,
        "education" : "Bachelors",
        "education-num" : 13,
        "marital-status" : "Married-civ-spouse",
        "occupation" : "Prof-specialty",
        "relationship" : "Husband",
        "race" : "Asian-Pac-Islander",
        "sex" : "Male",
        "capital-gain" : 0,
        "capital-loss" : 0,
        "hours-per-week" : 40,
        "native-country" : "India"
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200, \
        f"Response code is not successful. {response.json()}"
    assert response.json() == {"prediction":">50K"}