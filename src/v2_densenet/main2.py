import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Define paths
train_path = r"D:\ED\braintumor\data\Training"
test_path = r"D:\ED\braintumor\data\Testing"

# Image parameters
img_size = (128, 128)

# Function to load and preprocess images
def load_images_from_folder(folder, classes=['notumor', 'tumor']):
    images = []
    labels = []
    for label, subfolder in enumerate(classes):
        path = os.path.join(folder, subfolder)
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            if img_path.endswith('.jpg'):  # Ensure only image files are processed
                img = load_img(img_path, target_size=img_size)
                img = img_to_array(img) / 255.0  # Normalize pixel values
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Load training and test data
train_images, train_labels = load_images_from_folder(train_path)
test_images, test_labels = load_images_from_folder(test_path)

# Split train data into train and validation sets (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Load DenseNet121 pretrained model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze pretrained layers
base_model.trainable = False

# Add custom layers
x = GlobalAveragePooling2D()(base_model.output)  # Convert feature maps into vector
x = Dense(128, activation='relu')(x)  # Fully connected layer
x = Dropout(0.5)(x)  # Dropout for regularization
x = Dense(1, activation='sigmoid')(x)  # Binary classification output (Tumor / No Tumor)

# Create the model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predict on a random test image
random_idx = random.randint(0, len(test_images) - 1)
random_img = test_images[random_idx]
true_label = test_labels[random_idx]

# Predict
predicted_label = model.predict(random_img[np.newaxis, ...])[0][0]
predicted_label = 1 if predicted_label > 0.5 else 0

# Display results
plt.imshow(random_img)
plt.title(f"True: {'Tumor' if true_label == 1 else 'No Tumor'} | Predicted: {'Tumor' if predicted_label == 1 else 'No Tumor'}")
plt.axis('off')
plt.show()
