import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.config as cf
from src.models.SVM import SVM
from src.feature_engineering.encoder import OneHotEncoder
from src.feature_engineering.hog import HOG
from src.image_processing import gray_image_data, image_resize, white_padding_to_square
import random
import pickle as pkl

# Data path
folder_path = cf.TRAIN_DATA_PATH

# Get data
data, labels = gray_image_data(folder_path)

# Encode ouput labels
encoder = OneHotEncoder()
encoded_labels = encoder.fit_transform(labels)

# Save encoder
with open("WindowsSliding_HOG_SVM/models/vocab.pkl", "wb") as file:
    pkl.dump(encoder, file)

# Shuffle data
data_label = list(zip(data, encoded_labels))
random.shuffle(data_label)
data, encoded_labels = zip(*data_label)
data = list(data)
encoded_labels = list(encoded_labels)

# Resize data
data_padded = white_padding_to_square(data)
data_resized = image_resize(data_padded, cf.MODEL_INPUT_IMAGE_SIZE)

# Feature engineering
data_extracted = HOG(data_resized)

# Model
svm_classifier = SVM(encoder.vocab_size, data_extracted[0].shape[0])
svm_classifier.fit(data_extracted, encoded_labels, epochs=cf.EPOCHS, lr=cf.LEARNING_RATE, stop=cf.STOP_THRESHOLD)

# Test
print(svm_classifier.predict(data_extracted[0:10]))
print(encoder.reverse_transform(svm_classifier.predict_label(data_extracted[0:10])))
print(encoder.reverse_transform(encoded_labels)[0:10])

# Save
with open("WindowsSliding_HOG_SVM/models/svm_classifier.pkl", "wb") as file:
    pkl.dump(svm_classifier, file)
