import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'checking_balance':7000, 'months_loan_duration':10, 'age':70, 'employment_length':20, 'amount':20000,'Procedures':1})

print(r.json())
