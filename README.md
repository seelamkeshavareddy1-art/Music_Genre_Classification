# 🎵 Music Genre Classification using CNN

A deep learning project that predicts the **genre of a music audio file** using **MFCC (Mel-Frequency Cepstral Coefficients)** features and a **Convolutional Neural Network (CNN)**.

The project includes model training, command-line prediction, a Jupyter Notebook, and a Streamlit web application for uploading audio and viewing the predicted genre with a confidence score.

---

## 📌 Project Objective

The main objective of this project is to automatically classify music into one of **10 genres** based on audio characteristics.

### Supported Genres

- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Librosa | Audio loading and MFCC feature extraction |
| NumPy | Numerical and array operations |
| Scikit-learn | Data splitting and label encoding |
| TensorFlow / Keras | CNN model development and training |
| Matplotlib | Training graph and visualization |
| Streamlit | Web application |
| SoundFile | Audio file handling |
| Jupyter Notebook | Experimentation and documentation |

---

## 📂 Project Structure

```text
Music_Genre_Classification/
│
├── dataset/
│   ├── blues/
│   ├── classical/
│   ├── country/
│   ├── disco/
│   ├── hiphop/
│   ├── jazz/
│   ├── metal/
│   ├── pop/
│   ├── reggae/
│   └── rock/
│
├── models/
│   └── music_genre_cnn.keras
│
├── notebooks/
│   └── Music_Genre_Classification.ipynb
│
├── app.py
├── train.py
├── predict.py
├── features.py
├── download_dataset.py
├── training_accuracy.png
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🎧 Dataset

This project uses a **GTZAN-based 10-second audio dataset**.

For the current demonstration, the project contains:

- **100 audio clips**
- **10 genres**
- **10 clips per genre**
- **80% training data**
- **20% validation data**

The dataset is organized into separate folders according to genre.

```text
dataset/
├── blues/
├── classical/
├── country/
├── disco/
├── hiphop/
├── jazz/
├── metal/
├── pop/
├── reggae/
└── rock/
```

> This project uses a small demonstration subset rather than the complete GTZAN dataset.

---

## 🔍 How the System Works

The overall workflow is:

```text
Audio File
    ↓
Audio Loading
    ↓
MFCC Feature Extraction
    ↓
Feature Padding / Formatting
    ↓
CNN Model
    ↓
Softmax Classification
    ↓
Predicted Music Genre
    ↓
Confidence Score
```

### 1. Audio Loading

The audio file is loaded using **Librosa** with a sample rate of 22,050 Hz.

### 2. MFCC Feature Extraction

MFCC features are extracted from each audio file.

The project uses:

- Sample Rate: `22050`
- Number of MFCCs: `40`
- Maximum MFCC length: `130`

### 3. CNN Classification

The MFCC features are provided to a CNN containing convolution, pooling, dense, and dropout layers.

### 4. Prediction

The final Softmax layer produces probabilities for all 10 genres.

The genre with the highest probability is selected as the predicted genre.

---

## 🧠 CNN Architecture

The model contains:

```text
Input MFCC
    ↓
Conv2D - 32 filters
    ↓
MaxPooling2D
    ↓
Conv2D - 64 filters
    ↓
MaxPooling2D
    ↓
Conv2D - 128 filters
    ↓
MaxPooling2D
    ↓
Flatten
    ↓
Dense - 128 neurons
    ↓
Dropout - 0.3
    ↓
Softmax Output - 10 genres
```

### Model Configuration

- Optimizer: `Adam`
- Loss Function: `Categorical Crossentropy`
- Metric: `Accuracy`
- Epochs: `20`
- Batch Size: `32`

---

## 📊 Results

Using the current 100-audio demonstration dataset, the trained model achieved:

**Validation Accuracy: 70.00%**

The trained model is saved as:

```text
models/music_genre_cnn.keras
```

### Important Note

The current dataset is intentionally small for demonstration purposes. Therefore, the **70% validation accuracy should not be interpreted as the expected performance on the complete GTZAN dataset**.

The model also shows signs of overfitting on this small dataset, so using more training data and techniques such as data augmentation could improve generalization.

---

## 🚀 Installation

### 1. Clone or download the project

Open the project folder in VS Code.

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

On Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

If you want to train the model again:

```bash
python train.py
```

After training, the model will be saved at:

```text
models/music_genre_cnn.keras
```

The training accuracy graph will also be generated as:

```text
training_accuracy.png
```

> The already-trained model does not need to be retrained to use the prediction application.

---

## 🎯 Command-Line Prediction

You can classify an audio file directly from the terminal.

Example:

```bash
python predict.py dataset\blues\blues_000.wav
```

Example output:

```text
Loading trained model...
Model loaded successfully!
Processing audio...

==============================
     MUSIC GENRE RESULT
==============================
Predicted Genre : BLUES
Confidence      : 99.98%
==============================
```

The confidence value represents the probability assigned by the model to the predicted class.

---

## 🌐 Streamlit Web Application

The project includes a simple web interface where users can upload an audio file and classify its genre.

### Start the application

```bash
python -m streamlit run app.py
```

The Streamlit application provides:

- 🎵 Audio file upload
- ▶️ Audio playback
- 🎯 Genre prediction
- 📈 Confidence score
- 📊 Genre probability chart

### Supported Upload Formats

- `.wav`
- `.mp3`
- `.ogg`

---

## 📓 Jupyter Notebook

The project includes:

```text
notebooks/Music_Genre_Classification.ipynb
```

The notebook demonstrates:

1. Library imports
2. Dataset configuration
3. MFCC extraction
4. Dataset loading
5. Label encoding
6. Train-validation split
7. CNN model creation
8. Model training
9. Model evaluation
10. Accuracy visualization
11. Model saving
12. Audio prediction
13. MFCC visualization

---

## 📁 Main Files

### `features.py`

Contains the MFCC feature extraction function used to convert audio into model-ready features.

### `train.py`

Loads the dataset, extracts features, trains the CNN, evaluates it, saves the model, and creates the accuracy graph.

### `predict.py`

Loads the trained model and predicts the genre of an audio file from the command line.

### `app.py`

Runs the Streamlit web application for interactive audio upload and genre prediction.

### `download_dataset.py`

Used for obtaining the selected GTZAN-based demonstration audio samples.

### `Music_Genre_Classification.ipynb`

Provides the complete notebook-based workflow.

---

## 🔮 Future Improvements

Possible improvements include:

- Use the complete GTZAN dataset
- Increase the number of training samples
- Apply audio data augmentation
- Add spectrogram-based CNN classification
- Improve the CNN architecture
- Add Early Stopping and learning-rate scheduling
- Compare CNN with other machine learning models
- Deploy the Streamlit application online
- Add more detailed audio visualizations

---

## ⚠️ Limitations

- The current demonstration uses only 100 audio clips.
- The dataset is much smaller than the complete GTZAN collection.
- The model may overfit because of the small training set.
- Confidence scores are model probabilities and should not be interpreted as guaranteed correctness.
- Performance can vary depending on the audio quality and genre of an uploaded track.

---

## 👨‍💻 Project Summary

**Music Genre Classification using CNN** demonstrates how deep learning can be applied to audio classification.

The system extracts MFCC features from music files and uses a CNN to classify them into 10 music genres. A Streamlit interface makes the trained model easy to test by allowing users to upload an audio file and receive a predicted genre with its confidence score.

---

## 📜 License

This project is intended for educational and academic purposes.
