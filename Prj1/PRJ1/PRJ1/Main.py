from mytorch import Tensor
from mytorch.optimizer import SGD
from mytorch.loss import CategoricalCrossEntropy, MeanSquaredError
from mytorch.activation import softmax, relu
from mytorch.layer import Linear
from mytorch import Model
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
np.random.seed(126)

# Define the MLP model
class MLP(Model):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 8)
        self.fc2 = Linear(8, 16)
        self.fc3 = Linear(16, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = softmax(self.fc3(x))
        return x

# Load and Hadle Test Train Data
Train = pd.read_csv("./Iris-Train.csv")
Test  = pd.read_csv("./Iris-Test.csv")

X_train = Tensor(Train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].to_numpy())
X_test = Tensor(Test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].to_numpy())

Y_train = Train['Species'].to_numpy()
Y_test  = Test['Species'].to_numpy()

label_encoder = LabelEncoder()
Y_train = Tensor(label_encoder.fit_transform(Y_train).reshape(-1, 1) + 1)
Y_test = Tensor(label_encoder.transform(Y_test).reshape(-1, 1) + 1)

# Training Specs
model = MLP()
loss_function = MeanSquaredError
optimizer = SGD(model.parameters(), learning_rate=0.001)

epochs = 100
batch_size = 32

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
        y_batch.data = y_batch.data.reshape(batch_size, 1)
        
        # Compute loss
        loss = loss_function(predictions, y_batch)
        optimizer.zero_grad()
        grad = np.ones(loss.shape)
        loss.backward(grad)
        
        total_loss += loss.data.sum()
        
        # Optimizer step
        optimizer.step()

print(f"Training Finished, Loss: {total_loss / num_batches}")
print(f"Model Params:\n{np.round(model.fc1.weight.data, 1)}")
print(f"Model Params:\n{np.round(model.fc2.weight.data, 1)}")
print(f"Model Params:\n{np.round(model.fc3.weight.data, 1)}")