import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    """Multi-layer perceptron with ReLU activations"""

    def __init__(self, input_dim, hidden_sizes, output_dim, activation=nn.ReLU):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class KoopmanOperator(nn.Module):
    """Koopman Operator with learnable A, B, C matrices"""

    def __init__(
        self, state_dim, action_dim, latent_dim, encoder_hidden_sizes=[64, 64]
    ):
        super(KoopmanOperator, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Encoder: maps states to latent space
        self.encoder = MLP(state_dim, encoder_hidden_sizes, latent_dim)

        # Koopman matrices
        self.A = nn.Parameter(
            torch.randn(latent_dim, latent_dim) * 0.1
        )  # State transition
        self.B = nn.Parameter(
            torch.randn(action_dim, latent_dim) * 0.1
        )  # Control input
        self.C = nn.Parameter(torch.randn(latent_dim, state_dim) * 0.1)  # Observation

        # Data normalization parameters (will be set during training)
        self.register_buffer("state_shift", torch.zeros(state_dim))
        self.register_buffer("state_scale", torch.ones(state_dim))
        self.register_buffer("action_shift", torch.zeros(action_dim))
        self.register_buffer("action_scale", torch.ones(action_dim))

    def normalize_states(self, states):
        """Normalize states using stored shift and scale"""
        return (states - self.state_shift) / self.state_scale

    def normalize_actions(self, actions):
        """Normalize actions using stored shift and scale"""
        return (actions - self.action_shift) / self.action_scale

    def denormalize_states(self, states):
        """Denormalize states back to original scale"""
        return states * self.state_scale + self.state_shift

    def encode(self, states):
        """Encode states to latent space"""
        normalized_states = self.normalize_states(states)
        return self.encoder(normalized_states)

    def decode(self, latent):
        """Decode latent representation back to state space"""
        return torch.matmul(latent, self.C)

    def forward_step(self, latent, action):
        """Single step forward prediction in latent space"""
        normalized_action = self.normalize_actions(action)
        # z_{t+1} = A @ z_t + B^T @ u_t
        next_latent = torch.matmul(latent, self.A) + torch.matmul(
            normalized_action, self.B
        )
        return next_latent

    def forward(self, states, actions):
        """
        Forward pass through the Koopman operator

        Args:
            states: (batch_size, seq_len, state_dim) - state sequences
            actions: (batch_size, seq_len-1, action_dim) - action sequences

        Returns:
            predictions: (batch_size, seq_len-1, state_dim) - predicted next states
            latent_states: (batch_size, seq_len, latent_dim) - encoded latent states
        """

        batch_size, seq_len, _ = states.shape

        # Encode all states to latent space
        latent_states = self.encode(states.reshape(-1, self.state_dim))
        latent_states = latent_states.reshape(batch_size, seq_len, self.latent_dim)

        predictions = []

        # Iterative prediction
        for t in range(seq_len - 1):
            current_latent = latent_states[:, t]  # (batch_size, latent_dim)
            current_action = actions[:, t]  # (batch_size, action_dim)

            # Predict next latent state
            next_latent = self.forward_step(current_latent, current_action)

            # Decode to state space
            next_state = self.decode(next_latent)
            predictions.append(next_state)

        predictions = torch.stack(
            predictions, dim=1
        )  # (batch_size, seq_len-1, state_dim)

        return predictions, latent_states

    def multi_step_prediction(self, initial_state, actions):
        """
        Multi-step prediction from initial state

        Args:
            initial_state: (batch_size, state_dim) - initial state
            actions: (batch_size, horizon, action_dim) - action sequence

        Returns:
            predictions: (batch_size, horizon, state_dim) - predicted states
        """
        batch_size, horizon, _ = actions.shape

        # Encode initial state
        latent = self.encode(initial_state)  # (batch_size, latent_dim)

        predictions = []

        for t in range(horizon):
            # Predict next latent state
            latent = self.forward_step(latent, actions[:, t])

            # Decode to state space
            state = self.decode(latent)

            predictions.append(state)

        predictions = torch.stack(
            predictions, dim=1
        )  # (batch_size, horizon, state_dim)

        return predictions

    def compute_loss(self, states, actions, loss_weights=None):
        """
        Compute reconstruction loss

        Args:
            states: (batch_size, seq_len, state_dim)
            actions: (batch_size, seq_len-1, action_dim)
            loss_weights: (state_dim,) - weights for different state dimensions

        Returns:
            loss: scalar tensor
        """
        predictions, _ = self.forward(states, actions)
        targets = states[:, 1:, :]  # Target next states

        # Compute MSE loss
        # loss = F.mse_loss(predictions, targets, reduction="none")
        loss = F.l1_loss(predictions, targets, reduction="none")
        # loss = F.smooth_l1_loss(predictions, targets, reduction="none")

        # Apply loss weights if provided
        if loss_weights is not None:
            loss_weights = loss_weights.to(predictions.device)
            mse_loss = loss * loss_weights.unsqueeze(0).unsqueeze(0)

        return loss.mean()

    def set_normalization_params(
        self, state_shift, state_scale, action_shift, action_scale
    ):
        """Set normalization parameters"""
        self.state_shift.copy_(torch.tensor(state_shift, dtype=torch.float32))
        self.state_scale.copy_(torch.tensor(state_scale, dtype=torch.float32))
        self.action_shift.copy_(torch.tensor(action_shift, dtype=torch.float32))
        self.action_scale.copy_(torch.tensor(action_scale, dtype=torch.float32))
