from solvers.base_solver import BaseODESolver
from typing import Tuple
import torch
from torch import nn


def _rk45_get_states(initial_state: torch.Tensor, dynamics_function: nn.Module, t: torch.Tensor, step_size: float) -> torch.Tensor:

    state = initial_state
    batch_size = initial_state.shape[0]

    time_tensor = t.repeat(batch_size, 1)
    state_time_tensor = torch.cat((time_tensor, state), dim=1)
    k1 = dynamics_function(state_time_tensor)

    time_tensor = time_tensor + step_size * 2 / 9
    state_tensor = state + step_size * 2 / 9 * k1
    state_time_tensor = torch.cat((time_tensor, state_tensor), dim=1)
    k2 = dynamics_function(state_time_tensor)

    time_tensor = time_tensor + step_size / 3
    state_tensor = state + step_size * (k1 / 12 + k2 / 4)
    state_time_tensor = torch.cat((time_tensor, state_tensor), dim=1)
    k3 = dynamics_function(state_time_tensor)

    time_tensor = time_tensor + step_size * 3 / 4
    state_tensor = state + step_size * (k1 * 69 / 128 + k2 * -243 / 128 + k3 * 135 / 64)
    state_time_tensor = torch.cat((time_tensor, state_tensor), dim=1)
    k4 = dynamics_function(state_time_tensor)

    time_tensor = time_tensor + step_size
    state_tensor = state + step_size * (k1 * -17 / 12 + k2 * 27 / 4 + k3 * -27 / 5 + k4 * 16 / 15)
    state_time_tensor = torch.cat((time_tensor, state_tensor), dim=1)
    k5 = dynamics_function(state_time_tensor)

    time_tensor = time_tensor + step_size * 5 / 6
    state_tensor = state + step_size * (k1 * 65 / 432 + k2 * -5 / 16 + k3 * 13 / 16 + k4 * 4 / 27 + k5 * 5 / 144)
    state_time_tensor = torch.cat((time_tensor, state_tensor), dim=1)
    k6 = dynamics_function(state_time_tensor)

    return [k1, k2, k3, k4, k5, k6]


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

            k1, _, k3, k4, k5, k6 = _rk45_get_states(state, dynamics_function, t, self._step_size)

            state = state + k1 * 47 / 450 + k3 * 12 / 25 + k4 * 32 / 225 + k5 / 30 + k6 * -6 / 25
            states[i] = state

        return states
    

class AdaptiveRK45Solver(BaseODESolver):
    """
    Runge-Kutta-Fehlberg solver for ODEs with adaptive step size. Solves ODEs batchwise.
    """
    def __init__(self, time_range: Tuple[float, float], max_step_size: float, min_step_size: float, error_tolerance: float = 1e-6):
        super(AdaptiveRK45Solver, self).__init__()
        self._time_range = time_range
        self._max_step_size = torch.tensor(max_step_size)
        self._min_step_size = torch.tensor(min_step_size)
        self._error_tolerance = torch.tensor(error_tolerance)

        self._step_size = self._max_step_size

    def solve(self, initial_state: torch.Tensor, dynamics_function: nn.Module) -> torch.Tensor:
        state = initial_state
        batch_size = initial_state.shape[0]

        # The length of the states array is unknown, so we need to initialize it with a large enough number
        states = torch.zeros(10000, batch_size, *initial_state.shape[1:])
        states[0] = initial_state

        current_time = torch.tensor(self._time_range[0])
        current_index = 0

        step_size = torch.tensor(self._step_size)

        while current_time < self._time_range[1]:
            k1, _, k3, k4, k5, k6 = _rk45_get_states(state, dynamics_function, current_time, step_size)

            error = torch.abs(k1 / 150 + k3 * -3 / 100 + k4 * 16 / 75 + k5 / 20 + k6 * -6 / 25)
            new_step_size = 0.9 * step_size * (self._error_tolerance / error) ** (1 / 5)

            if error > self._error_tolerance:
                step_size = new_step_size
                continue

            else:
                step_size = new_step_size

            step_size = torch.clamp(step_size, min=self._min_step_size, max=self._max_step_size)

            state = state + k1 * 47 / 450 + k3 * 12 / 25 + k4 * 32 / 225 + k5 / 30 + k6 * -6 / 25
            
            states[current_index] = state
            current_index += 1
            current_time += step_size
