from flask import Flask, render_template, request
import pickle
import requests


app = Flask(__name__)

saved_model = open("model.pkl", "rb")
saved_vect = open("vect.pkl", "rb")

loaded_model = pickle.load(saved_model)
loaded_vect = pickle.load(saved_vect)


API_URL = "https://api-inference.huggingface.co/models/finiteautomata/bertweet-base-sentiment-analysis"
headers = {"Authorization": "Bearer"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


@app.route('/', methods=['POST', 'GET'])
def hello():
    predClickbaitAns = ""
    if request.method == "POST":
        text = request.form["clickbaitText"]
        if len(text.strip()) != 0:
            vect_text = loaded_vect.transform([text])
            pred = loaded_model.predict(vect_text)
            if pred[0] == 0:
                predClickbaitAns = "This is non-clickbait"
            else:
                predClickbaitAns = "This is clickbait"
    return render_template("index.html", clickbaitAns=predClickbaitAns)


@app.route("/sentiment", methods=["POST", "GET"])
def sentiment():
    pos = 0
    neg = 0
    if request.method == "POST":
        sent = request.form["sentimentText"]
        predSentimentAns = query({"inputs": sent})[0]
        pos = predSentimentAns[-1]["score"]
        neg = predSentimentAns[0]["score"]
    return render_template("sentiment.html", posAns=pos, negAns=neg)


if __name__ == "__main__":
    app.run(debug=True)