from solvers.base_solver import BaseODESolver
from typing import Tuple
import torch
from torch import nn


def _rk45_get_states(initial_state: torch.Tensor, dynamics_function: nn.Module, t: torch.Tensor, step_size: float, include_state_in_dynamics: bool = False) -> torch.Tensor:

    state = initial_state

    # Break up singletons
    time_tensor = t.unsqueeze(1) 
    step_size = step_size.unsqueeze(1)

    if include_state_in_dynamics:
        state_time_tensor = torch.cat((time_tensor, state), dim=1)
    else:
        state_time_tensor = time_tensor

    k1 = dynamics_function(state_time_tensor)

    time_tensor = time_tensor + step_size * 2 / 9
    state_tensor = state + step_size * 2 / 9 * k1

    if include_state_in_dynamics:
        state_time_tensor = torch.cat((time_tensor, state_tensor), dim=1)
    else:
        state_time_tensor = time_tensor

    k2 = dynamics_function(state_time_tensor)

    time_tensor = time_tensor + step_size / 3
    state_tensor = state + step_size * (k1 / 12 + k2 / 4)

    if include_state_in_dynamics:
        state_time_tensor = torch.cat((time_tensor, state_tensor), dim=1)
    else:
        state_time_tensor = time_tensor

    k3 = dynamics_function(state_time_tensor)

    time_tensor = time_tensor + step_size * 3 / 4
    state_tensor = state + step_size * (k1 * 69 / 128 + k2 * -243 / 128 + k3 * 135 / 64)

    if include_state_in_dynamics:
        state_time_tensor = torch.cat((time_tensor, state_tensor), dim=1)
    else:
        state_time_tensor = time_tensor

    k4 = dynamics_function(state_time_tensor)

    time_tensor = time_tensor + step_size
    state_tensor = state + step_size * (k1 * -17 / 12 + k2 * 27 / 4 + k3 * -27 / 5 + k4 * 16 / 15)

    if include_state_in_dynamics:
        state_time_tensor = torch.cat((time_tensor, state_tensor), dim=1)
    else:
        state_time_tensor = time_tensor

    k5 = dynamics_function(state_time_tensor)

    time_tensor = time_tensor + step_size * 5 / 6
    state_tensor = state + step_size * (k1 * 65 / 432 + k2 * -5 / 16 + k3 * 13 / 16 + k4 * 4 / 27 + k5 * 5 / 144)

    if include_state_in_dynamics:
        state_time_tensor = torch.cat((time_tensor, state_tensor), dim=1)
    else:
        state_time_tensor = time_tensor

    k6 = dynamics_function(state_time_tensor)

    return [k1, k2, k3, k4, k5, k6]


class RK45Solver(BaseODESolver):
    """
    Runge-Kutta-Fehlberg solver for ODEs. Solves ODEs batchwise.
    """
    def __init__(self, timesteps_number: int, time_range: Tuple[float, float], include_state_in_dynamics: bool = False):
        super(RK45Solver, self).__init__()
        self._timesteps_number = timesteps_number
        self._time_range = time_range

        self._timesteps = torch.linspace(*self._time_range, self._timesteps_number)
        self._step_size = self._timesteps[1] - self._timesteps[0]
        self._include_state_in_dynamics = include_state_in_dynamics

    def solve(self, initial_state: torch.Tensor, dynamics_function: nn.Module) -> torch.Tensor:
        state = initial_state
        batch_size = initial_state.shape[0]
        step_size_tensor = self._step_size.repeat(batch_size)

        states = torch.zeros(batch_size, self._timesteps_number,  *initial_state.shape[1:])
        states[:, 0] = initial_state

        # todo: Include table of coefficients as a parameter
        for i, t in enumerate(self._timesteps):

            k1, _, k3, k4, k5, k6 = _rk45_get_states(state, dynamics_function, t.repeat(batch_size), step_size_tensor, self._include_state_in_dynamics)

            state = state + k1 * 47 / 450 + k3 * 12 / 25 + k4 * 32 / 225 + k5 / 30 + k6 * -6 / 25
            states[:, i] = state

        return states
    

class AdaptiveRK45Solver(BaseODESolver):
    """
    Runge-Kutta-Fehlberg solver for ODEs with adaptive step size. Solves ODEs batchwise.
    """
    def __init__(self, time_range: Tuple[float, float], max_step_size: float, min_step_size: float, error_tolerance: float = 1e-3, include_state_in_dynamics: bool = False):
        super(AdaptiveRK45Solver, self).__init__()
        self._time_range = time_range
        self._max_step_size = torch.tensor(max_step_size)
        self._min_step_size = torch.tensor(min_step_size)
        self._error_tolerance = torch.tensor(error_tolerance)

        self._step_size = self._max_step_size
        self._buffer_size = int(self._time_range[-1] // self._min_step_size) + 1
        self._include_state_in_dynamics = include_state_in_dynamics

    def solve(self, initial_state: torch.Tensor, dynamics_function: nn.Module) -> torch.Tensor:
        state = initial_state.clone()
        batch_size = initial_state.shape[0]
        
        states = torch.zeros(batch_size, self._buffer_size, *initial_state.shape[1:])
        states[:, 0] = initial_state
        
        current_times = torch.tensor(self._time_range[0], dtype=torch.float).repeat(batch_size)
        current_indices = torch.zeros(batch_size, dtype=torch.int)
        step_sizes_tensor = torch.tensor(self._step_size, dtype=torch.float).repeat(batch_size)

        active_ode_mask = torch.ones(batch_size, dtype=torch.bool)
        
        while active_ode_mask.any():

            active_mask = active_ode_mask.clone()
            active_states = state[active_mask]
            active_step_sizes = step_sizes_tensor[active_mask]
            active_times = current_times[active_mask]

            k1, _, k3, k4, k5, k6 = _rk45_get_states(active_states, dynamics_function, active_times, active_step_sizes, self._include_state_in_dynamics)

            errors = torch.norm(k1 / 150 + k3 * -3 / 100 + k4 * 16 / 75 + k5 / 20 + k6 * -6 / 25, dim=1)
            new_step_sizes = 0.9 * active_step_sizes * (self._error_tolerance / errors) ** (1 / 5)
            
            step_sizes_tensor[active_mask] = torch.clamp(new_step_sizes, min=self._min_step_size, max=self._max_step_size)

            repeat_mask = errors > self._error_tolerance
            continue_mask = ~repeat_mask
            
            if continue_mask.any():

                new_states = (active_states[continue_mask] +
                            k1[continue_mask] * 47 / 450 +
                            k3[continue_mask] * 12 / 25 +
                            k4[continue_mask] * 32 / 225 +
                            k5[continue_mask] / 30 +
                            k6[continue_mask] * -6 / 25)
                
                active_indices = torch.where(active_mask)[0]
                update_batch_indices = active_indices[continue_mask]
                state_indices_to_update = current_indices[update_batch_indices]

                states[update_batch_indices, state_indices_to_update] = new_states
                state[update_batch_indices] = new_states
                current_times[update_batch_indices] += active_step_sizes[continue_mask]
                current_indices[update_batch_indices] += 1
                
                completed_mask = current_times[update_batch_indices] >= self._time_range[1]
                active_ode_mask[update_batch_indices[completed_mask]] = False

        return states[:, :current_indices.max()]
