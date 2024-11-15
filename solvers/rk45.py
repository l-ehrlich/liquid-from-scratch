from solvers.base_solver import BaseODESolver
from typing import Tuple
import torch
from torch import nn


class RK45Solver(BaseODESolver):
    """
    Runge-Kutta-Fehlberg solver for ODEs. Solves ODEs batchwise.
    """
    def __init__(self, timesteps_number: int, time_range: Tuple[float, float]):
        super(RK45Solver, self).__init__()
        self._timesteps_number = timesteps_number
        self._time_range = time_range

        self._timesteps = torch.linspace(*self._time_range, self._timesteps_number)
        self._step_size = self._timesteps[1] - self._timesteps[0]

    def solve(self, initial_state: torch.Tensor, dynamics_function: nn.Module) -> torch.Tensor:
        state = initial_state
        batch_size = initial_state.shape[0]

        states = torch.zeros(self._timesteps_number, batch_size, *initial_state.shape[1:])
        states[0] = initial_state

        # todo: Include table of coefficients as a parameter
        for i, t in enumerate(self._timesteps):

            time_tensor = t.repeat(batch_size, 1)
            state_time_tensor = torch.cat((time_tensor, state), dim=1)
            k1 = dynamics_function(state_time_tensor)

            time_tensor = time_tensor + self._step_size * 2 / 9
            state_tensor = state + self._step_size * 2 / 9 * k1
            state_time_tensor = torch.cat((time_tensor, state_tensor), dim=1)
            k2 = dynamics_function(state_time_tensor)

            time_tensor = time_tensor + self._step_size / 3
            state_tensor = state + self._step_size * (k1 / 12 + k2 / 4)
            state_time_tensor = torch.cat((time_tensor, state_tensor), dim=1)
            k3 = dynamics_function(state_time_tensor)

            time_tensor = time_tensor + self._step_size * 3 / 4
            state_tensor = state + self._step_size * (k1 * 69 / 128 + k2 * -243 / 128 + k3 * 135 / 64)
            state_time_tensor = torch.cat((time_tensor, state_tensor), dim=1)
            k4 = dynamics_function(state_time_tensor)

            time_tensor = time_tensor + self._step_size
            state_tensor = state + self._step_size * (k1 * -17 / 12 + k2 * 27 / 4 + k3 * -27 / 5 + k4 * 16 / 15)
            state_time_tensor = torch.cat((time_tensor, state_tensor), dim=1)
            k5 = dynamics_function(state_time_tensor)

            time_tensor = time_tensor + self._step_size * 5 / 6
            state_tensor = state + self._step_size * (k1 * 65 / 432 + k2 * -5 / 16 + k3 * 13 / 16 + k4 * 4 / 27 + k5 * 5 / 144)
            state_time_tensor = torch.cat((time_tensor, state_tensor), dim=1)
            k6 = dynamics_function(state_time_tensor)

            state = state + k1 * 47 / 450 + k3 * 12 / 25 + k4 * 32 / 225 + k5 / 30 + k6 * -6 / 25
            states[i] = state

        return states
