import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# 1. Feature Engineering with NumPy
np.random.seed(42)
n_samples = 1500

usage_hours = np.random.uniform(5, 120, n_samples)
support_tickets = np.random.randint(0, 10, n_samples)
contract_months = np.random.choice([1, 6, 12, 24], n_samples)
monthly_bill = np.random.uniform(30, 300, n_samples)

# Business logic: high tickets + high bill - long contract = churn
churn_prob = (support_tickets * 0.15) + (monthly_bill / 500) - (contract_months / 40)
churn = (churn_prob > 0.45).astype(int)

X = np.column_stack([usage_hours, support_tickets, contract_months, monthly_bill])
y = churn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 2. PyTorch Deep Learning Classifier
class ChurnModel(nn.Module):
    def __init__(self, input_dim):
        super(ChurnModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))

model = ChurnModel(input_dim=4)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. Training Loop
for epoch in range(60):
    optimizer.zero_grad()
    preds = model(X_train_t)
    loss = criterion(preds, y_train_t)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    test_preds = model(X_test_t)
    acc = (test_preds > 0.5).eq(y_test_t).sum().item() / len(y_test_t)
    print(f"Model Trained successfully. Test Accuracy: {acc * 100:.2f}%")

torch.save(model.state_dict(), "churn_model.pth")
print("Saved artifacts: churn_model.pth, scaler.pkl")