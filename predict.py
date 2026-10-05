import sys
import numpy as np
from tensorflow.keras.models import load_model
from features import extract_mfcc

MODEL_PATH = "models/music_genre_cnn.keras"

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

print("Loading trained model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")


def predict_genre(audio_file):

    print("Processing audio...")

    mfcc = extract_mfcc(audio_file)

    if mfcc is None:
        print("Could not process audio file.")
        return

    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    probabilities = model.predict(mfcc, verbose=0)[0]

    predicted_index = np.argmax(probabilities)

    predicted_genre = GENRES[predicted_index]

    confidence = probabilities[predicted_index] * 100

    print()
    print("==============================")
    print("     MUSIC GENRE RESULT")
    print("==============================")
    print("Predicted Genre :", predicted_genre.upper())
    print(f"Confidence      : {confidence:.2f}%")
    print("==============================")
    print()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please provide an audio file.")
        print()
        print("Example:")
        print(r"python predict.py dataset\blues\blues_000.wav")
    else:
        predict_genre(sys.argv[1])