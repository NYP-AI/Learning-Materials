import requests

API_URL = "https://api-inference.huggingface.co/models/finiteautomata/bertweet-base-sentiment-analysis"
headers = {"Authorization": "Bearer"}


def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

sent = "This is the user input"
output = query({"inputs": sent})[0]
pos = output[-1]["score"]
neg = output[0]["score"]