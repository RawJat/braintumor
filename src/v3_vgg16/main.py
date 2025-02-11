import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Define paths for training and testing data
train_path = r"D:\ED\braintumor\data\Training"
test_path = r"D:\ED\braintumor\data\Testing"

# Image dimensions (VGG16 requires 224x224)
img_size = (224, 224)

# Function to load images and labels from a given folder
def load_images_from_folder(folder):
    images = []
    labels = []
    for label, subfolder in enumerate(['notumor', 'tumor']):  # Assign numeric labels (0: No Tumor, 1: Tumor)
        path = os.path.join(folder, subfolder)
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            if img_path.endswith('.jpg'):  # Process only JPG images
                img = load_img(img_path, target_size=img_size)  # Resize image
                img = img_to_array(img) / 255.0  # Normalize pixel values
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Load the dataset
train_images, train_labels = load_images_from_folder(train_path)
test_images, test_labels = load_images_from_folder(test_path)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Load VGG16 pre-trained model (without the top classification layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the existing VGG16 layers to use them as a feature extractor
for layer in base_model.layers:
    layer.trainable = False

# Adding custom layers on top of VGG16 for our classification task
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)  # Binary classification (0 or 1)

# Define the final model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model with Adam optimizer and binary cross-entropy loss
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with training data and validate with validation data
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Select a random image from the test set for prediction
random_idx = random.randint(0, len(test_images) - 1)
random_img = test_images[random_idx]
true_label = test_labels[random_idx]

# Make a prediction
predicted_label = model.predict(random_img[np.newaxis, ...])[0][0]
predicted_label = 1 if predicted_label > 0.5 else 0

# Display the image along with the actual and predicted labels
plt.imshow(random_img)
plt.title(f"Actual: {'Tumor' if true_label == 1 else 'No Tumor'} | Predicted: {'Tumor' if predicted_label == 1 else 'No Tumor'}")
plt.axis('off')
plt.show()
