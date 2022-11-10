from main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greeting": "Hello World!"}

def test_inference():
    test_data = \
    {
        "age" : 37,
        "workclass" : "Private",
        "fnlgt" : 280464,
        "education" : "Some-college",
        "education-num" : 10,
        "marital-status" : "Married-civ-spouse",
        "occupation" : "Exec-managerial",
        "relationship" : "Husband",
        "race" : "Black",
        "sex" : "Male",
        "capital-gain" : 0,
        "capital-loss" : 0,
        "hours-per-week" : 80,
        "native-country" : "United-States"
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200, \
        f"Response code is not successful. {response.json()}"
    assert response.json() == {"prediction":">50K"}