import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("data.csv")

# texts = df["headline"].values
# labels = df["clickbait"].values
#
# vect = CountVectorizer()
# vect.fit(texts)
# vect_texts = vect.transform(texts)
#
# model = LogisticRegression()
# # (Input, Output)
# model.fit(vect_texts, labels)
#
# sent = "Carbon emissions rise by nine percent"
# vect_sent = vect.transform([sent])
# pred = model.predict(vect_sent)
# if pred[0] == 0:
#     print("This is Non-Clickbait")
# else:
#     print("This is Clickbait")
#
# saved_model = open("model.pkl", "wb")
# saved_vect = open("vect.pkl", "wb")
#
# # This is to save
# pickle.dump(model, saved_model)
# pickle.dump(vect, saved_vect)

saved_model = open("model.pkl", "rb")
saved_vect = open("vect.pkl", "rb")

loaded_model = pickle.load(saved_model)
loaded_vect = pickle.load(saved_vect)

sent = "Carbon emissions rise by nine percent"
vect_sent = loaded_vect.transform([sent])
pred = loaded_model.predict(vect_sent)
if pred[0] == 0:
    print("This is Non-Clickbait")
else:
    print("This is Clickbait")