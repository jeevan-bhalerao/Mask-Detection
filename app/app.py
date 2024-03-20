from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2

app = Flask(__name__)

# Load the trained model
model = load_model('mask_detector_model.h5')

# Function to predict mask on single image
def predict_mask(image):
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
    return label

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the POST request has an image attached
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in the request'}), 400

    image_file = request.files['image']

    # Read image file from request
    image_np = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Perform prediction on the image
    result = predict_mask(image)

    return jsonify({'result': result}), 200

if __name__ == '__main__':
    app.run(debug=True)
