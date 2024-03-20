
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained model
model = load_model('mask_detector_model.h5')

# Function to predict mask on single image
def predict_mask(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    orig_image = image.copy()  # Make a copy of the original image for display
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)[0][0]
    if pred > 0.5:
        label = "Mask"
    else:
        label = "No Mask"
    confidence = pred if pred > 0.5 else 1 - pred
    print(f"Predicted Label: {label}, Confidence: {confidence:.2f}")
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    cv2.putText(orig_image, label, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.imshow('Image', orig_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the test image
image_path = '/Users/LIFE/Documents/Projects/object_detection_project/20170922_174403.jpg'

# Perform prediction on the test image
predict_mask(image_path)
