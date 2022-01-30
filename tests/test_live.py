import requests

url = "https://udacity-nd0821.herokuapp.com/predict"

payload = {
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

r = requests.post(url, json=payload)

print(f'Status code: {r.status_code}')
assert r.status_code == 200
print(f'Response body: {r.json()}')
assert r.json() == {"prediction": "<=50K"}
