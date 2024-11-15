import torch
from torch import nn
from solvers.base_solver import BaseODESolver
from typing import Tuple


class EulerSolver(BaseODESolver):
    """
    Euler solver for ODEs. Solves ODEs batchwise.
    """
    def __init__(self, timesteps_number: int, time_range: Tuple[float, float]):
        super(EulerSolver, self).__init__()
        self._timesteps_number = timesteps_number
        self._time_range = time_range

        self._timesteps = torch.linspace(*self._time_range, self._timesteps_number)
        self._step_size = self._timesteps[1] - self._timesteps[0]

    def solve(self, initial_state: torch.Tensor, dynamics_function: nn.Module) -> torch.Tensor:
        state = initial_state
        batch_size = initial_state.shape[0]

        states = torch.zeros(self._timesteps_number, batch_size, *initial_state.shape[1:])
        states[0] = initial_state

        for i, t in enumerate(self._timesteps):
            state_time_tensor = torch.cat((t.repeat(batch_size, 1), state), dim=1)
            state = state + self._step_size * dynamics_function(state_time_tensor)
            states[i] = state

        return states
