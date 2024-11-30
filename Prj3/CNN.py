from mytorch import Tensor
from mytorch.optimizer import SGD
from mytorch.loss import CategoricalCrossEntropy
from mytorch.activation import softmax, relu
from mytorch.layer import Conv2d, MaxPool2d, Linear
from mytorch import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
import numpy as np


# Define the CNN model
class CNN(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool1 = MaxPool2d(in_channels=16, out_channels=16, kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool2 = MaxPool2d(in_channels=32, out_channels=32, kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = Linear(64 * 32 * 8 * 8, 128)  # 7x7 is the spatial size after pooling
        self.fc2 = Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = relu(self.conv1(x))
        x = self.pool1(x)
        x = relu(self.conv2(x))
        x = self.pool2(x)
        x.data = x.data.flatten()
        x = relu(self.fc1(x))
        x = softmax(self.fc2(x))
        return x


# Load and handle MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
data = mnist.data / 255.0  # Normalize to [0, 1]
labels = mnist.target.astype(int)

# Reshape data to (N, 1, 28, 28) for CNN
X = data.to_numpy().reshape(-1, 1, 28, 28)
Y = labels.to_numpy()

# Split into training and test sets
split_index = 60000
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

# Convert to Tensor
X_train = Tensor(X_train)
X_test = Tensor(X_test)
Y_train = Tensor(Y_train.reshape(-1, 1) + 1)  # +1 for compatibility with loss
Y_test = Tensor(Y_test.reshape(-1, 1) + 1)

# Training specs
model = CNN()
loss_function = CategoricalCrossEntropy
optimizer = SGD(model.parameters(), learning_rate=0.001)

epochs = 10
batch_size = 64

# Training loop
for epoch in range(epochs):
    total_loss = 0
    num_batches = X_train.shape[0] // batch_size

    for i in range(num_batches):
        # Get the batch
        start = i * batch_size
        end = start + batch_size
        X_batch = X_train[start:end]
        y_batch = Y_train[start:end]

        # Forward pass
        predictions = model(X_batch)

        # Compute loss
        loss = loss_function(predictions, y_batch)
        total_loss += loss.data.sum()

        # Optimizer step
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / num_batches}")

# Test the model
correct = 0
total = X_test.shape[0]
predictions = model(X_test)
predicted_labels = predictions.data.argmax(axis=1).reshape(-1, 1)
correct = np.sum(predicted_labels == (Y_test.data - 1))  # -1 to revert label shift
accuracy = correct / total

print(f"Test Accuracy: {accuracy * 100:.2f}%")
