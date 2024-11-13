import torch.nn as nn
import torch
from .adjoint import odeint_euler_adjoint

class NeuralODEClassifier(nn.Module):
    def __init__(self, ode_func, num_classes=3):
        super(NeuralODEClassifier, self).__init__()
        self.ode_func = ode_func
        self.classifier = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, y0, t):
        # Integrate ODE using the adjoint method
        y = odeint_euler_adjoint(self.ode_func, y0, t)
        # Use the final state for classification
        out = self.classifier(y[-1])
        return out
