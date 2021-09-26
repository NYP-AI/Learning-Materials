import streamlit as st
import cv2
import numpy as np
from PIL import Image


st.header("Face Detection 2.0")
st.text("Provide an image and we'll detect the faces")

cls = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def detect(PIL_img):
    cv2_img = cv2.cvtColor(np.array(PIL_img), cv2.COLOR_RGB2BGR)
    cv2_img_bw = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    boxes = cls.detectMultiScale(cv2_img_bw, 1.3, 5)
    for box in boxes:
        x, y, width, height = box
        cv2.rectangle(cv2_img, (x, y), (x + width, y + height), (255, 0, 0), 2)

    st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    st.markdown(f"Detected {len(boxes)} faces")


uploaded_file = st.file_uploader("Upload image here....", type=["jpeg", "jpg", "png"])
if uploaded_file is not None:
    PIL_img = Image.open(uploaded_file)
    detect(PIL_img)