import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Define paths
train_path = r"D:\ED\braintumor\data\Training"
test_path = r"D:\ED\braintumor\data\Testing"
model_path = "vgg16_braintumor.keras"  # Model save path

# Image size required for VGG16
img_size = (224, 224)

# Function to load images and labels from a folder
def load_images_from_folder(folder):
    images = []
    labels = []
    for label, subfolder in enumerate(['notumor', 'tumor']):  # Assign labels (0: No Tumor, 1: Tumor)
        path = os.path.join(folder, subfolder)
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            if img_path.endswith('.jpg'):
                img = load_img(img_path, target_size=img_size)  # Resize image
                img = img_to_array(img) / 255.0  # Normalize
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Load dataset
train_images, train_labels = load_images_from_folder(train_path)
test_images, test_labels = load_images_from_folder(test_path)

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Check if a trained model exists
if os.path.exists(model_path):
    print("Loading pre-trained model...")
    model = load_model(model_path)
else:
    print("No saved model found. Training a new model...")

    # Load VGG16 pre-trained model (without the top classification layer)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze VGG16 layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)  # Binary classification

    # Define final model
    model = Model(inputs=base_model.input, outputs=x)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

    # Save the trained model
    model.save(model_path)
    print(f"Model saved as {model_path}")

# Evaluate model on test dataset
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Select a random test image for prediction
random_idx = random.randint(0, len(test_images) - 1)
random_img = test_images[random_idx]
true_label = test_labels[random_idx]

# Make a prediction
predicted_label = model.predict(random_img[np.newaxis, ...])[0][0]
predicted_label = 1 if predicted_label > 0.5 else 0

# Display the image along with actual and predicted labels
plt.imshow(random_img)
plt.title(f"Actual: {'Tumor' if true_label == 1 else 'No Tumor'} | Predicted: {'Tumor' if predicted_label == 1 else 'No Tumor'}")
plt.axis('off')
plt.show()
