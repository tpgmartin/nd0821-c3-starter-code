from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_say_hello():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}


def test_predict_positive():
    item = {
        'age': 20,
        'workclass': 'Private',
        'fnlgt': 27049,
        'education': 'Some-college',
        'education-num': 10,
        'marital-status': 'Never-married',
        'occupation': 'Exec-managerial',
        'relationship': 'Own-child',
        'race': 'White',
        'sex': 'Female',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 20,
        'native-country': 'United-States',
        'salary': '<=50K'
    }
    r = client.post("/predict", json=item)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


def test_predict_negative():
    item = {
        'age': 41,
        'workclass': 'Local-gov',
        'fnlgt': 297248,
        'education': 'Prof-school',
        'education-num': 15,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Prof-specialty',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 2415,
        'hours-per-week': 45,
        'native-country': 'United-States',
        'salary': '>50K'
    }
    r = client.post("/predict", json=item)
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}
