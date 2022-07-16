# importing the libraries 
from flask import Flask, render_template, request
import pickle



# loading the models 

loaded_model = open("Send to Participants/Pickle files/model.pkl", "rb")
loaded_vect = open("Send to Participants/Pickle files/vect.pkl", "rb")

model = pickle.load(loaded_model)
vectorizer = pickle.load(loaded_vect)


#using the api for sentiment analysis 
import requests

API_URL = "https://api-inference.huggingface.co/models/finiteautomata/bertweet-base-sentiment-analysis"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "I like you. I love you",
})

print(output)

# #main app route
app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def hello():
    spamAns = ""
    if request.method == "POST":
        text = request.form["spamText"]
        sent_vect = vectorizer.transform([text])
        pred = model.predict(sent_vect)
        if pred == "spam":
            spamAns = "This is spam"
        elif pred == "ham":
            spamAns = "This is not spam"
    return render_template('index.html', spamAns=spamAns)

# creating another app route for sentiment analysis 
@app.route("/sentiment", methods=["POST", "GET"])
def sentiment():
    pos_output = 0
    neg_output = 0
    sentimentAns = ""
    if request.method == "POST":
        text = request.form["sentimentText"]
        sentimentAns = query({"inputs": text})
        pos_output += round(sentimentAns[0][2]["score"] * 100, 2)
        neg_output += round(sentimentAns[0][0]["score"] * 100, 2)
    return render_template("sentiment.html", sentimentAns=sentimentAns, pos_output = pos_output, neg_output = neg_output )


if __name__ == '__main__':
      app.run(debug=True, port="8080")