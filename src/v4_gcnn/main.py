import os
import numpy as np
import tensorflow as tf
import spektral
from spektral.layers import GCNConv
from spektral.data.loaders import SingleLoader
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

# Set Image Paths
train_path = r"D:\ED\braintumor\data\Training"
test_path = r"D:\ED\braintumor\data\Testing"

# Image Parameters
img_size = (32, 32)  # Reduce size to optimize memory
num_channels = 3  # RGB channels

# Function to Convert Images into Graphs (Using Sparse Adjacency Matrices)
def load_images_as_graph(folder):
    images, labels, adjacency_matrices = [], [], []

    for label, subfolder in enumerate(['notumor', 'tumor']):
        path = os.path.join(folder, subfolder)
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            if img_path.endswith('.jpg'):
                # Load and normalize image
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
                img = tf.keras.preprocessing.image.img_to_array(img) / 255.0

                # Flatten image into node features
                img_flat = img.reshape(-1, num_channels)  # Each pixel is a node
                num_nodes = img_flat.shape[0]

                # Sparse adjacency matrix (identity with slight connections)
                adjacency_matrix = np.eye(num_nodes) + np.random.normal(0, 0.01, (num_nodes, num_nodes))
                adjacency_matrix = csr_matrix(adjacency_matrix)  # Convert to sparse format

                images.append(img_flat)
                labels.append(label)
                adjacency_matrices.append(adjacency_matrix)

    return np.array(images), np.array(labels), adjacency_matrices  # Keep adjacency matrices sparse

# Load Data
train_images, train_labels, train_adj = load_images_as_graph(train_path)
test_images, test_labels, test_adj = load_images_as_graph(test_path)

# Split Train Data into Training and Validation
X_train, X_val, y_train, y_val, A_train, A_val = train_test_split(
    train_images, train_labels, train_adj, test_size=0.2, random_state=42
)

# Define GCNN Model
class BrainTumorGCNN(Model):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCNConv(32, activation='relu')
        self.gcn2 = GCNConv(64, activation='relu')
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dropout = Dropout(0.5)
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x, a = inputs  # x: Node features, a: Adjacency matrix
        x = self.gcn1([x, a])
        x = self.gcn2([x, a])
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.output_layer(x)

# Convert adjacency matrices to Spektral format
train_adj = [spektral.utils.normalized_adjacency(a).astype('float32') for a in train_adj]
val_adj = [spektral.utils.normalized_adjacency(a).astype('float32') for a in A_val]
test_adj = [spektral.utils.normalized_adjacency(a).astype('float32') for a in test_adj]

# Compile GCNN Model
gcnn_model = BrainTumorGCNN()
gcnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
gcnn_model.fit(
    x=[X_train, np.array(train_adj)], y=y_train,
    validation_data=([X_val, np.array(val_adj)], y_val),
    batch_size=8,  # Reduce batch size for memory efficiency
    epochs=5
)

# Evaluate on Test Data
test_loss, test_accuracy = gcnn_model.evaluate([test_images, np.array(test_adj)], test_labels, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
