import pytest
from webapp.heart_disease_webapp import app


@pytest.fixture
def client():
    return app.test_client()


def test_get(client):
    response = client.get("/")
    assert "200 OK" in response.status
    assert b"Heart Disease Analysis with Machine Learning Model" in response.data


def test_post(client):
    response = client.post("/predictions", data={
        "age": 34,
        "sex": 1,
        "cp": 0,
        "trestbps": 120,
        "chol": 124,
        "fbs": 0,
        "restecg": 1,
        "thalach": 130,
        "exang": 1,
        "oldpeak": 0.2,
        "slope": 2,
        "ca": 0,
        "thal": 3
    })
    assert b"There is 45.0% chance of having a heart disease." in response.data


def test_missing_values(client):
    response = client.post("/predictions", data={
        "age": 23,
        "sex": 1
    })
    assert b"Predicting not possible, all features needed." in response.data
