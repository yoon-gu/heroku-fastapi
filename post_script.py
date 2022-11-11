'''
Script for POST request on Heroku APP
'''

import requests

input_data = {
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


response = requests.post('https://yoon-gu-udacity.herokuapp.com/predict/',
                          json=input_data,
                          timeout=10)

print("response status code: ", response.status_code)
print("response: ", response.json())