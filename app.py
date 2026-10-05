import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from features import extract_mfcc

st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="centered"
)

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]

MODEL_PATH = "models/music_genre_cnn.keras"


@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)


st.title("🎵 Music Genre Classification")
st.write("Upload an audio file and the CNN model will predict its music genre.")

st.divider()

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Please run train.py first.")
    st.stop()

model = load_trained_model()

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "ogg"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    temp_file = "temp_audio.wav"

    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🎯 Predict Genre"):

        with st.spinner("Analyzing audio..."):

            mfcc = extract_mfcc(temp_file)

            if mfcc is None:
                st.error("Unable to process audio.")
            else:

                mfcc = mfcc[np.newaxis, ..., np.newaxis]

                probabilities = model.predict(
                    mfcc,
                    verbose=0
                )[0]

                predicted_index = np.argmax(probabilities)

                predicted_genre = GENRES[predicted_index]

                confidence = probabilities[predicted_index] * 100

                st.success(
                    f"Predicted Genre: {predicted_genre.upper()}"
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.subheader("Genre Probabilities")

                probability_data = {
                    GENRES[i]: float(probabilities[i] * 100)
                    for i in range(len(GENRES))
                }

                st.bar_chart(probability_data)

    if os.path.exists(temp_file):
        os.remove(temp_file)