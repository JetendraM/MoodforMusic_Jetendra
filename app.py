import urllib.request
import streamlit as st
import pickle
import cv2
import pandas as pd
import joblib
import numpy as np
CNN_Model=joblib.load("Emotion_detection_model")

# Download the face detection model
url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
filename = "haarcascade_frontalface_default.xml"
urllib.request.urlretrieve(url, filename)
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the music player data
Music_Player = pd.read_csv(r"C:\Users\jeten\Documents\Capstone Project\Trial and error\DataScience_Diaries\Final Capstone Project\MoodforMusic\data_moods.csv")
Music_Player = Music_Player[["name", "artist", "mood", "popularity"]]

# Function to recommend songs based on predicted mood
def Recommend_Songs(pred_class):
    mood_map = {
        "Disgust": "Sad",
        "Happy": "Happy",
        "Sad": "Happy",
        "Fear": "Calm",
        "Angry": "Calm",
        "Surprise": "Energetic",
        "Neutral": "Energetic"
    }
    mood = mood_map.get(pred_class, "Happy")  # Default to "Happy" if pred_class not found
    Play = Music_Player[Music_Player["mood"] == mood]
    Play = Play.sort_values(by="popularity", ascending=False)
    Play = Play[:5].reset_index(drop=True)
    return Play

# Function to load and prepare an image
def load_and_prep_image(uploaded_file, img_shape=255):
    img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    if img is None:
        st.error(f"Error: Unable to load image at path {filename}")
        return None
    
    GrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(GrayImg, 1.1, 4)
    for x, y, w, h in faces:
        roi_GrayImg = img[y: y + h, x: x + w]
        roi_Img = img[y: y + h, x: x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    RGBImg = cv2.resize(RGBImg, (img_shape, img_shape))
    RGBImg = RGBImg / 255.0
    return RGBImg

# Function to predict and plot the result
def pred_and_plot(uploaded_file, class_names):
    img = load_and_prep_image(uploaded_file)
    if img is None:
        return
    
    pred = CNN_Model.predict(np.expand_dims(img, axis=0))
    pred_class = class_names[pred.argmax()]
    
    st.image(img, caption=f"Prediction: {pred_class}")
    Play = Recommend_Songs(pred_class)
    st.dataframe(Play)

# Streamlit interface
st.title("Emotion-based Music Recommendation")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
class_names = ["Disgust", "Happy", "Sad", "Fear", "Angry", "Surprise", "Neutral"]

if uploaded_file is not None:
    pred_and_plot(uploaded_file, class_names)
