from .ODE_function import ODEFunc
from .NODE_classifier import NeuralODEClassifier
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
np.random.seed(42)

def generate_spiral_data(points_per_class, num_classes, noise=0.4):
    """
    Generates a spiral dataset.

    Args:
        points_per_class (int): Number of points per class.
        num_classes (int): Number of spiral classes.
        noise (float): Standard deviation of Gaussian noise added to the data.

    Returns:
        X (np.ndarray): Feature matrix of shape (points_per_class * num_classes, 2).
        y (np.ndarray): Labels of shape (points_per_class * num_classes,).
    """
    X = []
    y = []
    for class_number in range(num_classes):
        ix = range(points_per_class * class_number, points_per_class * (class_number + 1))
        r = np.linspace(0.0, 1, points_per_class)  # Radius
        theta = np.linspace(class_number * 4, (class_number + 1) * 4, points_per_class) + np.random.randn(points_per_class) * noise
        X_class = np.c_[r * np.sin(theta * 2.5), r * np.cos(theta * 2.5)]
        X.append(X_class)
        y += [class_number] * points_per_class
    X = np.concatenate(X)
    y = np.array(y)
    return X, y


# Parameters for the spiral dataset
points_per_class = 1500
num_classes = 2  # Increase the number of classes for added complexity
noise = 0.4

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Generate the spiral data
X, y = generate_spiral_data(points_per_class, num_classes, noise)

t0, t1 = 0.0, 1.0
num_steps = 40
t = torch.linspace(t0, t1, steps=num_steps)

num_epochs = 100
batch_size = 64
loss_history = []
accuracy_history = []

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

ode_func = ODEFunc()
model = NeuralODEClassifier(ode_func, num_classes=2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_func = torch.nn.CrossEntropyLoss