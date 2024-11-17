import torch
from torch import nn
from typing import Literal
from solvers.base_solver import BaseODESolver


class NeuralODENetwork(nn.Module):
    def __init__(self, latent_dynamics_function: nn.Module, solver: BaseODESolver, gradient_method: Literal['adjoint', 'backprop'] = 'backprop'):
        super(NeuralODENetwork, self).__init__()

        self._latent_dynamics_function = latent_dynamics_function
        self._pre_solver_function = None
        self._task_head = None
        self._gradient_method = gradient_method
        self._solver = solver

    def attach_pre_solver_function(self, pre_solver_function: nn.Module):
        self._pre_solver_function = pre_solver_function

    def attach_task_head(self, task_head: nn.Module):
        self._task_head = task_head

    def forward(self, initial_state: torch.Tensor, ):
        if self._pre_solver_function is not None:
            initial_state = self._pre_solver_function(initial_state)

        hidden_states = self._solver.solve(initial_state, self._latent_dynamics_function)

        last_hidden_state = hidden_states[-1]

        if self._task_head is not None:
            return self._task_head(last_hidden_state)

        return last_hidden_state
