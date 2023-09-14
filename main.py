import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Data Preparation
data_dir = 'path_to_speech_commands_dataset'
word_list = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

# Data Preprocessing
def extract_mfcc_features(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000)  # Sample rate may vary
    mfccs = librosa.feature.mfcc(audio, sr=16000, n_mfcc=13)
    return mfccs

X, y = [], []

for word in word_list:
    word_dir = os.path.join(data_dir, word)
    for filename in os.listdir(word_dir):
        if filename.endswith(".wav"):
            audio_path = os.path.join(word_dir, filename)
            mfccs = extract_mfcc_features(audio_path)
            X.append(mfccs.T)  # Transpose for consistent sequence length
            y.append(word_list.index(word))

X = np.array(X)
y = np.array(y)

# Splitting into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Model Architecture
model = models.Sequential([
    layers.Input(shape=(None, 13)),  # MFCC features with 13 coefficients
    layers.LSTM(64, return_sequences=True),
    layers.Dense(len(word_list) + 1, activation='softmax')  # +1 for the blank label in CTC loss
])

# Training
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

