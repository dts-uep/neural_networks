import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.config as cf
from src.feature_engineering.hog import HOG
from src.image_processing import gray_image_test, image_resize
import pickle as pkl

# Get validate data
data = gray_image_test(cf.VAL_DATA_PATH)

# Get classifier
with open("WindowsSliding_HOG_SVM/models/svm_classifier.pkl", "rb") as file:
    svm_classifier = pkl.load(file)

# Feature engineering
data = image_resize(data, cf.MODEL_INPUT_IMAGE_SIZE)
data = HOG(data)

# Validate model
print(svm_classifier.predict(data))
result = svm_classifier.predict_label(data)

# Decode
with open("WindowsSliding_HOG_SVM/models/encoder.pkl", "rb") as file:
    encoder = pkl.load(file)

print(encoder.reverse_transform(result)) 
print(os.listdir(cf.VAL_DATA_PATH))

