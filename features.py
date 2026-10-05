import librosa
import numpy as np


SAMPLE_RATE = 22050
DURATION = 30
N_MFCC = 40
MAX_PAD_LEN = 130


def extract_mfcc(file_path):

    try:

        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            duration=DURATION
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC
        )

        # Make every audio file the same size
        if mfcc.shape[1] < MAX_PAD_LEN:

            pad_width = MAX_PAD_LEN - mfcc.shape[1]

            mfcc = np.pad(
                mfcc,
                pad_width=(
                    (0, 0),
                    (0, pad_width)
                ),
                mode="constant"
            )

        else:

            mfcc = mfcc[:, :MAX_PAD_LEN]

        return mfcc

    except Exception as e:

        print(
            f"Error processing {file_path}: {e}"
        )

        return None