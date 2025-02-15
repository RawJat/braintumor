import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Debugging function
def debug(msg, var):
    print(f"{msg}: {var}")


# Load dataset (assuming adjacency matrices and node features are precomputed)
def load_brain_tumor_graph_data(folder):
    graphs = []
    labels = []

    for label, subfolder in enumerate(['notumor', 'tumor']):
        path = os.path.join(folder, subfolder)
        for filename in os.listdir(path):
            if filename.endswith('.npz'):  # Assume graph data stored in .npz
                data = np.load(os.path.join(path, filename))
                x = torch.tensor(data['features'], dtype=torch.float)  # Node features
                edge_index = torch.tensor(data['edges'], dtype=torch.long)  # Graph structure
                y = torch.tensor([label], dtype=torch.long)  # Label

                graphs.append(Data(x=x, edge_index=edge_index.t().contiguous(), y=y))
                labels.append(label)

    return graphs, np.array(labels)


# Paths
train_path = r"D:\ED\braintumor\data\Training"
test_path = r"D:\ED\braintumor\data\Testing"

# Load graph dataset
train_graphs, train_labels = load_brain_tumor_graph_data(train_path)
test_graphs, test_labels = load_brain_tumor_graph_data(test_path)

# Split into train and validation sets
train_graphs, val_graphs = train_test_split(train_graphs, test_size=0.2, random_state=42)

# DataLoader
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)


# Define Graph Convolutional Neural Network (GCNN)
class BrainTumorGCNN(torch.nn.Module):
    def __init__(self):
        super(BrainTumorGCNN, self).__init__()
        self.conv1 = GCNConv(64, 128)  # Assuming 64 node features
        self.conv2 = GCNConv(128, 64)
        self.fc = torch.nn.Linear(64, 2)  # Binary classification (tumor vs. no tumor)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = torch.mean(x, dim=0)  # Global pooling (mean over nodes)
        x = self.fc(x)
        return F.log_softmax(x, dim=0)


# Model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
debug("Device", device)

model = BrainTumorGCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()


# Training loop
def train_model():
    model.train()
    for epoch in range(1, 11):  # 10 Epochs
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output.unsqueeze(0), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}")


# Evaluation
def evaluate(loader):
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            pred = output.argmax().item()
            pred_labels.append(pred)
            true_labels.append(batch.y.item())

    accuracy = accuracy_score(true_labels, pred_labels)
    return accuracy, true_labels, pred_labels


# Train the model
train_model()

# Evaluate on test set
test_accuracy, y_true, y_pred = evaluate(test_loader)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot predictions vs. true labels
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_true)), y_true, label="True Labels", marker="o", alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, label="Predicted Labels", marker="x", alpha=0.6)
plt.xlabel("Sample Index")
plt.ylabel("Class Label")
plt.title("True vs. Predicted Labels (GCNN)")
plt.legend()
plt.show()
