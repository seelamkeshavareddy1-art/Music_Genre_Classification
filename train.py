import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.utils import to_categorical

from features import extract_mfcc


# -----------------------------------
# Settings
# -----------------------------------

DATASET_PATH = "dataset"
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


# -----------------------------------
# Create model directory
# -----------------------------------

os.makedirs(
    "models",
    exist_ok=True
)


# -----------------------------------
# Load dataset
# -----------------------------------

X = []
y = []


print("Loading audio files...")


for genre in GENRES:

    genre_path = os.path.join(
        DATASET_PATH,
        genre
    )

    if not os.path.exists(genre_path):

        print(
            f"Warning: {genre_path} not found."
        )

        continue

    for filename in os.listdir(
        genre_path
    ):

        if filename.endswith(".wav"):

            file_path = os.path.join(
                genre_path,
                filename
            )

            mfcc = extract_mfcc(
                file_path
            )

            if mfcc is not None:

                X.append(mfcc)
                y.append(genre)


print(
    f"Total audio files loaded: {len(X)}"
)


# -----------------------------------
# Convert to NumPy arrays
# -----------------------------------

X = np.array(X)

y = np.array(y)


# CNN expects:
# samples, height, width, channels

X = X[..., np.newaxis]


print("Feature shape:", X.shape)


# -----------------------------------
# Encode labels
# -----------------------------------

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(
    y
)

y_encoded = to_categorical(
    y_encoded,
    num_classes=len(GENRES)
)


# -----------------------------------
# Train-test split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(
    "Training samples:",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_test)
)


# -----------------------------------
# Build CNN
# -----------------------------------

model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(
            X.shape[1],
            X.shape[2],
            1
        )
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        len(GENRES),
        activation="softmax"
    )
])


# -----------------------------------
# Compile model
# -----------------------------------

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


model.summary()


# -----------------------------------
# Train
# -----------------------------------

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_test,
        y_test
    ),

    epochs=20,

    batch_size=32
)


# -----------------------------------
# Evaluate
# -----------------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test
)


print(
    f"Validation Accuracy: {accuracy * 100:.2f}%"
)


# -----------------------------------
# Save model
# -----------------------------------

model.save(
    MODEL_PATH
)


print(
    f"Model saved to: {MODEL_PATH}"
)


# -----------------------------------
# Plot accuracy
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()

plt.savefig(
    "training_accuracy.png"
)

plt.show()