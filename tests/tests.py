import pytest
import json
from webApp.heart_disease_webApp import app


@pytest.fixture
def client():
    return app.test_client()


def test_get(client):
    response = client.get("/")
    assert "200 OK" in response.status
    assert b"Heart Disease Analysis with Machine Learning Model" in response.data


def test_post(client):
    response = client.post("/predictions", data={
        "age": 23,
         "sex": 1,
         "cp": 1,
         "trestbps": 120,
         "chol": 124,
         "fbs": 1,
         "restecg": 1,
         "thalach": 173,
         "exang": 0,
         "oldpeak": 1,
         "slope": 2,
         "ca": 1,
         "thal": 3
    })
    assert "200 OK" in response.status
    assert b"It is 1" in response.data
