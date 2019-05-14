import json

with open('Tokens.json','r') as f:
    data = json.load(f)

for key, value in data.items():
    print(key, value)
