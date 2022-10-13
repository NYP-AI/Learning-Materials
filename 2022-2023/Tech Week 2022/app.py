import streamlit as st
from transformers import pipeline 
import requests
from bs4 import BeautifulSoup

tab1, tab2 = st.tabs(["Link", "Text"])
summarizer = pipeline("summarization")


with tab1:
    st.title("Text Summary App 📚")
    st.header("Enter your Link")
    textLink = st.text_input("Enter your text below: ")
    if textLink != "":
        r = requests.get(textLink)
        soup = BeautifulSoup(r.text, features="html.parser")
        results = soup.find_all("p")

        text = ""
        for sent in results:
            text += sent.get_text()

        st.header("Summary")
        if text != "":
            hf_summary = summarizer(text, max_length = 400, min_length = 100, do_sample = False ,truncation=True)
            st.write(hf_summary[0]['summary_text'])



with tab2:
    st.title("Text Summary App 📚")
    st.header("Enter your text")
    text = st.text_area("Enter your text below: ")

    if text != "":
        hf_summary = summarizer(text, max_length=400, min_length=100, do_sample=False, truncation=True)
        st.write(hf_summary[0]['summary_text'])

