import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split

# Define paths
train_path = r"D:\ED\braintumor\data\Training"
test_path = r"D:\ED\braintumor\data\Testing"

# Image parameters
img_size = (128, 128)  # Resize images to 128x128

# Load and preprocess data
def load_images_from_folder(folder):
    images = []
    labels = []
    for label, subfolder in enumerate(['notumor', 'tumor']):
        path = os.path.join(folder, subfolder)
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            if img_path.endswith('.jpg'):  # Only process JPG files
                img = load_img(img_path, target_size=img_size)
                img = img_to_array(img) / 255.0
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

train_images, train_labels = load_images_from_folder(train_path)
test_images, test_labels = load_images_from_folder(test_path)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Apply data augmentation
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

# Model building function with hyperparameter tuning
def build_model(hp):
    model = keras.Sequential([
        keras.layers.Conv2D(
            filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
            kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
            activation='relu',
            input_shape=(img_size[0], img_size[1], 3)
        ),
        keras.layers.Conv2D(
            filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
            kernel_size=hp.Choice('conv_2_kernel', values=[3, 5]),
            activation='relu'
        ),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(
            units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
            activation='relu'
        ),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# Hyperparameter tuning
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    directory='output',
    project_name='Brain_Tumor_Detection'
)

tuner.search(train_generator, epochs=10, validation_data=val_generator)

# Get best model
best_model = tuner.get_best_models(num_models=1)[0]

# Model summary
best_model.summary()

# Evaluate on test data
test_loss, test_accuracy = best_model.evaluate(test_images, test_labels, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predict on a random test image
random_idx = random.randint(0, len(test_images) - 1)
random_img = test_images[random_idx]
true_label = test_labels[random_idx]

# Predict
predicted_label = best_model.predict(random_img[np.newaxis, ...])[0][0]
predicted_label = 1 if predicted_label > 0.5 else 0

# Display results
plt.imshow(random_img)
plt.title(f"True: {'Tumor' if true_label == 1 else 'No Tumor'} | Predicted: {'Tumor' if predicted_label == 1 else 'No Tumor'}")
plt.axis('off')
plt.show()
