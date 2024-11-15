import torch
from torch import nn
from abc import ABC, abstractmethod


class BaseODESolver(ABC):
    """
    Base class for ODE solvers. Responsible for managing the timesteps and solving the ODE.
    """
    def __init__(self):
        pass

    @abstractmethod
    def solve(self, initial_state: torch.Tensor, dynamics_function: nn.Module) -> torch.Tensor:
        pass
