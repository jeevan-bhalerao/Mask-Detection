import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Function to load dataset
def load_dataset(directory):
    images = []
    labels = []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            images.append(image_path)
            labels.append(label)
    return images, labels

# Load dataset
dataset_dir = 'Mask-detection/dataset'
images, labels = load_dataset(dataset_dir)

# Split dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create directories for train and test data
train_dir = 'train_data'
test_dir = 'test_data'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Move images to train and test directories
for image, label in zip(train_images, train_labels):
    label_dir = os.path.join(train_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    os.rename(image, os.path.join(label_dir, os.path.basename(image)))

for image, label in zip(test_images, test_labels):
    label_dir = os.path.join(test_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    os.rename(image, os.path.join(label_dir, os.path.basename(image)))

# Define image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Data augmentation and generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Define model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-4),
              metrics=['accuracy'])

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=2,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

# Evaluate model
loss, accuracy = model.evaluate(test_generator)
print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')

# Save model
model.save('mask_detector_model.h5')
