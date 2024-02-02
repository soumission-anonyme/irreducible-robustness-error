import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
import numpy as np
from scipy.integrate import dblquad
from joblib import Parallel, delayed

X, y = make_moons(n_samples=2000, noise=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1667, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = SimpleNN()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
test_accuracies = []

import csv, os

model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to('cuda' if torch.cuda.is_available() else 'cpu')
        y_batch = y_batch.to('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.round() == y_batch).float().sum()
    
    train_losses.append(total_loss / len(train_loader))
    train_accuracies.append(correct / len(train_loader.dataset))
    
    model.eval()
    X_val = X_val.to('cuda' if torch.cuda.is_available() else 'cpu')
    y_val = y_val.to('cuda' if torch.cuda.is_available() else 'cpu')
    val_outputs = model(X_val)
    val_loss = criterion(val_outputs, y_val).item()
    val_losses.append(val_loss)
    correct = (val_outputs.round() == y_val).float().sum()
    val_accuracies.append(correct / len(X_val))

    print(f"Epoch {epoch + 1}/{epochs} - Training Accuracy: {100 * train_accuracies[-1]:.2f}%\tValidation Accuracy: {100 * val_accuracies[-1]:.2f}%")

    csv_header = ["Epoch", "Train", "Validate"]
    csv_file_path = "train_accuracies.csv"
    write_header = not os.path.exists(csv_file_path)
    with open(csv_file_path, mode="a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        if write_header:
            csv_writer.writerow(csv_header)
        csv_writer.writerow([epoch, train_accuracies[-1].item(), val_accuracies[-1].item()])

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# Define the PDF functions for U1+G and U2+G
def p0(x1, x2, sigma = 0.3):

    # The constant multiplier
    constant = 1 / (2 * np.pi**2 * sigma**2)
    # The exponential term outside the integral
    exp_term = np.exp(-(x1**2 + x2**2 + 1) / (2 * sigma**2))
    # The integral term
    integral = numerical_part_p0(x1, x2, sigma)
    return constant * exp_term * integral

def p1(x1, x2, sigma = 0.3):

    # The constant multiplier
    constant = 1 / (2 * np.pi**2 * sigma**2)
    # The exponential term outside the integral
    exp_term = np.exp(-((x1 - 1)**2 + (x2 - 0.5)**2 + 1) / (2 * sigma**2))
    # The integral term
    integral = numerical_part_p1(x1, x2, sigma)
    return constant * exp_term * integral

# Define the convolution integral functions for U1+G and U2+G
def numerical_part_p0(x1, x2, sigma):

    integrand = lambda t: np.exp((x1*np.cos(t) + x2*np.sin(t)) / sigma**2)
    result, _ = integrate.quad(integrand, 0, np.pi)
    return result

def numerical_part_p1(x1, x2, sigma):

    integrand = lambda t: np.exp((- (x1 - 1) * np.cos(t) - (x2 - 0.5) * np.sin(t)) / sigma**2)
    result, _ = integrate.quad(integrand, 0, np.pi)
    return result


X1, X2 = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))

mesh = np.c_[X1.ravel(), X2.ravel()]
mesh = torch.tensor(mesh, dtype=torch.float32)
with torch.no_grad():
    Z = model(mesh.to('cuda' if torch.cuda.is_available() else 'cpu')).round()
Z = Z.reshape(X1.shape).cpu().numpy()

from scipy.integrate import simps

accuracy_values = np.vectorize(p0)(X1, X2) * (1 - Z) / 2 + np.vectorize(p1)(X1, X2) * Z / 2

integral_result = simps(simps(accuracy_values, np.linspace(-3, 3, 100), axis=0), np.linspace(-3, 3, 100))

print(f"Simpson's rule Integral Result: {integral_result:6f}, rough estimate: {accuracy_values.sum()* (X1[1,1]-X1[0,0])**2:6f}", )



import csv
correct__ = []
# correct_certified__ = []

n = 10000000
for k in range(100):
    X_, y_ = make_moons(n_samples=n, noise=0.3)
    X__ = torch.tensor(X_, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
    y__ = torch.tensor(y_, dtype=torch.float32).view(-1, 1).to('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        test_outputs = model(X__)
        correct_ = (test_outputs.round() == y__).float().sum().item()

        # X__bounded = BoundedTensor(X_, ptb)
        # lb, ub = model.compute_bounds(x=(X__bounded,), method="CROWN")
        # correct_certified_ = ((y_ == 1) & (lb > threshold)).sum() + ((y_ == 0) & (ub < threshold)).sum().item()

    correct__.append(correct_)
    # correct_certified__.append(correct_certified_)

    avg_correct__ = np.cumsum(correct__) / np.arange(1, len(correct__) + 1) / n
    # avg_correct_certified__ = np.cumsum(correct_certified__) / np.arange(1, len(correct_certified__) + 1)


    print(f"Batch {(k + 1)} - Correct: {correct_}/{n}, Test Accuracy (Sample Size {(k + 1)*n}): {avg_correct__[-1]:.6f}")
    
    
    csv_header = ["Step", "Correct", "Increment", "Accuracy"]
    csv_file_path = "accuracy_convergence.csv"
    write_header = not os.path.exists(csv_file_path)
    with open(csv_file_path, mode="a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        if write_header:
            csv_writer.writerow(csv_header)
        csv_writer.writerow([k + 1, correct_, n, avg_correct__[-1]])

model = model.to('cpu')
def accurate(x1, x2):
    with torch.no_grad():
        pred = model(torch.tensor([x1, x2])).round().item()
    return (p0(x1, x2) * (1 - pred) + p1(x1, x2) * pred) / 2

num_jobs = 16
x2_ranges = np.stack((np.linspace(-3, 3, num_jobs + 1)[:-1], np.linspace(-3, 3, num_jobs + 1)[1:]), axis=-1)

def dblquad_wrapper(x2_range):
    return dblquad(accurate, x2_range[0], x2_range[1], lambda x: -3, lambda x: 3)

results = Parallel(n_jobs=num_jobs, backend='multiprocessing')(delayed(dblquad_wrapper)(r) for r in x2_ranges)
expected_accuracy = sum(result[0] for result in results)
print("Result of the double integration:", expected_accuracy)