import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Load the data from the CSV
data = pd.read_csv('labeled_pose_data.csv')

# Print the first few rows to identify columns
print(data.head())

# Drop the column with filenames (e.g., 'image_path' or whatever it's called)
data = data.drop(columns=['image'])  # Replace 'image_path' with the actual column name

# Prepare the features and labels
X = data.iloc[:, :-1].values  # Features: all columns except the last (pose keypoints)
y = data.iloc[:, -1].values   # Labels: last column

# Step 2: Preprocess the data
# Encode the labels ("Badly injured", "Not injured or slightly injured", "Unknown") as numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # e.g., "Badly injured" -> 0, "Not injured" -> 1, "Unknown" -> 2

# Standardize the features (recommended for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Step 3: Create a PyTorch Dataset and DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# EXTRA STEP: RECORDING RESULTS FOR THE TEST VALUES
test_data_df = pd.DataFrame(X_test, columns=data.columns[:-1])  # Use original feature names
test_data_df['label'] = label_encoder.inverse_transform(y_test)  # Add the original labels

# Save the test data with labels to a new CSV file
test_data_df.to_csv('REZ.csv', index=False)

# Step 4: Define the neural network architecture
class PoseEstimationNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PoseEstimationNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

# Initialize the model
input_size = X_train.shape[1]  # 99 (for 33 key points with x, y, z coordinates)
hidden_size = 64               # You can adjust this value
output_size = 3                # 3 classes: "Badly injured", "Not injured", "Unknown"

model = PoseEstimationNN(input_size, hidden_size, output_size)

# Step 5: Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Train the model
num_epochs = 50

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize the weights

        running_loss += loss.item()
    
    # Print average loss for the epoch
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

# Step 7: Evaluate the model
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # Disable gradient computation for evaluation
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # Get class with the highest score
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy on the test set: {accuracy * 100:.2f}%')

