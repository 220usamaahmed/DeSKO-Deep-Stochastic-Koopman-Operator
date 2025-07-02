import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.utils.data import Dataset, DataLoader
import minari
import os
from koopman_operator import KoopmanOperator  # Assuming you saved the previous code
import matplotlib.pyplot as plt

DATASETS = {
    "InvertedDoublePendulum-v5": "mujoco/inverteddoublependulum/expert-v0",
    "InvertedPendulum-v5": "mujoco/invertedpendulum/expert-v0",
}

# ENV_NAME = "InvertedDoublePendulum-v5"
ENV_NAME = "InvertedPendulum-v5"

DATASET_NAME = DATASETS[ENV_NAME]

LATENT_DIM = 24
SEQ_LENGTH = 20


class TrajectoryDataset(Dataset):
    """Dataset for storing and loading trajectory data"""

    def __init__(self, states, actions, seq_len):
        """
        Args:
            states: (num_trajectories, traj_len, state_dim)
            actions: (num_trajectories, traj_len-1, action_dim)
            seq_len: Length of sequences to extract
        """
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
        self.seq_len = seq_len
        self.num_trajs, self.traj_len, _ = states.shape

        # Create valid starting indices for each trajectory
        self.valid_indices = []
        for traj_idx in range(self.num_trajs):
            for start_idx in range(self.traj_len - seq_len + 1):
                self.valid_indices.append((traj_idx, start_idx))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        traj_idx, start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.seq_len

        states = self.states[traj_idx, start_idx:end_idx]
        actions = self.actions[traj_idx, start_idx : end_idx - 1]

        return states, actions

    def save(self, save_dir):
        """Save states and actions as numpy arrays in the specified directory"""
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "states.npy"), self.states.numpy())
        np.save(os.path.join(save_dir, "actions.npy"), self.actions.numpy())

    @staticmethod
    def load(save_dir, seq_len):
        """Load states and actions from numpy arrays in the specified directory and create a TrajectoryDataset"""
        states_path = os.path.join(save_dir, "states.npy")
        actions_path = os.path.join(save_dir, "actions.npy")
        states = np.load(states_path)
        actions = np.load(actions_path)
        return TrajectoryDataset(states, actions, seq_len)


def collect_random_trajectories(env, num_trajectories=1000, max_steps=20):
    """Collect trajectories using random actions"""
    all_states = []
    all_actions = []

    for traj in range(num_trajectories):
        states = []
        actions = []

        state = env.reset()
        if isinstance(state, tuple):  # Handle new gym API
            state = state[0]
        states.append(state)

        for step in range(max_steps):
            # Take random action
            action = env.action_space.sample()

            # Step environment
            next_state, reward, done, truncated, info = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]

            actions.append(action)
            states.append(next_state)

            if done or truncated:
                break

        all_states.append(np.array(states))
        all_actions.append(np.array(actions))

        if (traj + 1) % 100 == 0:
            print(f"Collected {traj + 1}/{num_trajectories} trajectories")

    return all_states, all_actions


def collect_minari_trajectories(dataset, num_trajectories=1000, max_steps=20):
    all_states = []
    all_actions = []

    episode_iter = dataset.iterate_episodes()
    curr_episode = next(episode_iter)

    t = 0

    while True:
        if t > len(curr_episode):
            curr_episode = next(episode_iter)
            t = 0

        states = curr_episode.observations[t : t + max_steps]
        actions = curr_episode.actions[t : t + max_steps - 1]

        all_states.append(states)
        all_actions.append(actions)

        if len(all_states) == num_trajectories:
            break

    return all_states, all_actions


def compute_normalization_stats(states_list, actions_list):
    """Compute normalization statistics from trajectory data"""
    # Flatten all states and actions
    all_states = np.concatenate([s for s in states_list], axis=0)
    all_actions = np.concatenate([a for a in actions_list], axis=0)

    # Compute statistics
    state_mean = np.mean(all_states, axis=0)
    state_std = np.std(all_states, axis=0) + 1e-8  # Add small epsilon
    action_mean = np.mean(all_actions, axis=0)
    action_std = np.std(all_actions, axis=0) + 1e-8

    return state_mean, state_std, action_mean, action_std


def pad_trajectories(states_list, actions_list, max_len=None):
    """Pad trajectories to same length for batch processing"""
    if max_len is None:
        max_len = max(len(s) for s in states_list)

    state_dim = states_list[0].shape[1]
    action_dim = actions_list[0].shape[1]

    padded_states = []
    padded_actions = []

    for states, actions in zip(states_list, actions_list):
        traj_len = len(states)

        if traj_len > max_len:
            # Truncate if too long
            states = states[:max_len]
            actions = actions[: max_len - 1]
        else:
            # Pad if too short
            state_padding = np.zeros((max_len - traj_len, state_dim))
            action_padding = np.zeros((max_len - traj_len, action_dim))

            states = np.concatenate([states, state_padding], axis=0)
            actions = np.concatenate([actions, action_padding], axis=0)

        padded_states.append(states)
        padded_actions.append(actions)

    return np.array(padded_states), np.array(padded_actions)


def train_using_random_sampling(
    env_name="CartPole-v1",
    num_trajectories=1000,
    max_traj_len=20,
    seq_len=10,
    latent_dim=LATENT_DIM,
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-3,
):
    # Create environment
    env = gym.make(env_name)

    # Collect trajectories
    print("Collecting trajectories...")
    states_list, actions_list = collect_random_trajectories(
        env, num_trajectories, max_traj_len
    )

    train_koopman_model(
        env,
        states_list,
        actions_list,
        max_traj_len,
        seq_len,
        latent_dim,
        num_epochs,
        batch_size,
        learning_rate,
    )


def train_using_expert_dataset(
    dataset_name="mujoco/inverteddoublependulum/expert-v0",
    num_trajectories=1000,
    max_traj_len=20,
    seq_len=10,
    latent_dim=LATENT_DIM,
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-3,
):
    dataset = minari.load_dataset(dataset_name, download=True)
    env = dataset.recover_environment()

    print("Collecting trajectories...")
    states_list, actions_list = collect_minari_trajectories(
        dataset, num_trajectories=num_trajectories, max_steps=max_traj_len
    )

    train_koopman_model(
        env,
        states_list,
        actions_list,
        max_traj_len,
        seq_len,
        latent_dim,
        num_epochs,
        batch_size,
        learning_rate,
    )


def train_koopman_model(
    env,
    states_list,
    actions_list,
    max_traj_len,
    seq_len,
    latent_dim,
    num_epochs,
    batch_size,
    learning_rate,
):
    """Complete training pipeline"""

    state_dim = env.observation_space.shape[0]

    # Handle different action space types
    if hasattr(env.action_space, "n"):  # Discrete action space
        action_dim = env.action_space.n
        print(
            "Warning: Discrete action space detected. You may need to modify action sampling."
        )
    else:  # Continuous action space
        action_dim = env.action_space.shape[0]

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Compute normalization statistics
    print("Computing normalization statistics...")
    state_mean, state_std, action_mean, action_std = compute_normalization_stats(
        states_list, actions_list
    )

    # Pad trajectories to same length
    print("Padding trajectories...")
    states_array, actions_array = pad_trajectories(
        states_list, actions_list, max_traj_len
    )

    print(
        f"Final data shape - States: {states_array.shape}, Actions: {actions_array.shape}"
    )

    # Split into train/val
    split_idx = int(0.8 * len(states_array))
    train_states = states_array[:split_idx]
    train_actions = actions_array[:split_idx]
    val_states = states_array[split_idx:]
    val_actions = actions_array[split_idx:]

    # Create datasets and dataloaders
    train_dataset = TrajectoryDataset(train_states, train_actions, seq_len)
    val_dataset = TrajectoryDataset(val_states, val_actions, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    model = KoopmanOperator(state_dim, action_dim, latent_dim)
    model.set_normalization_params(state_mean, state_std, action_mean, action_std)

    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = float("inf")

    val_loss = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0

        for states, actions in train_loader:
            states, actions = states.to(device), actions.to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(states, actions)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for states, actions in val_loader:
                states, actions = states.to(device), actions.to(device)
                loss = model.compute_loss(states, actions)
                val_loss += loss.item()
                num_val_batches += 1

        val_loss /= num_val_batches

        # Update learning rate
        scheduler.step()

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            print("-" * 50)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "normalization_stats": {
                        "state_mean": state_mean,
                        "state_std": state_std,
                        "action_mean": action_mean,
                        "action_std": action_std,
                    },
                },
                "best_koopman_model.pth",
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": num_epochs,
            "val_loss": val_loss,
            "normalization_stats": {
                "state_mean": state_mean,
                "state_std": state_std,
                "action_mean": action_mean,
                "action_std": action_std,
            },
        },
        "final_koopman_model.pth",
    )

    print(f"Training completed! Best validation loss: {best_val_loss:.6f}")

    env.close()
    return model


def evaluate_model(model, env, num_episodes=10, max_steps=200):
    """Evaluate the trained model by comparing predictions with actual trajectories"""
    model.eval()
    device = next(model.parameters()).device

    prediction_errors = []

    with torch.no_grad():
        for episode in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]

            states = [state]
            actions = []

            # Collect a trajectory
            for step in range(max_steps):
                action = env.action_space.sample()
                next_state, _, done, truncated, _ = env.step(action)
                if isinstance(next_state, tuple):
                    next_state = next_state[0]

                actions.append(action)
                states.append(next_state)

                if done or truncated:
                    break

            if len(states) < 5:  # Need at least 5 steps for prediction
                continue

            # Convert to tensors
            states_tensor = (
                torch.FloatTensor(states).unsqueeze(0).to(device)
            )  # [1, T, state_dim]
            actions_tensor = (
                torch.FloatTensor(actions).unsqueeze(0).to(device)
            )  # [1, T-1, action_dim]

            # Make predictions
            predictions, _ = model(states_tensor, actions_tensor)

            # Compare with actual next states
            actual_next_states = states_tensor[:, 1:, :]
            error = torch.mean((predictions - actual_next_states) ** 2).item()
            prediction_errors.append(error)

    mean_error = np.mean(prediction_errors)
    std_error = np.std(prediction_errors)

    print(f"Prediction Error - Mean: {mean_error:.6f}, Std: {std_error:.6f}")
    return mean_error, std_error


def load_model(
    checkpoint_path="best_koopman_model.pth",
    state_dim=None,
    action_dim=None,
    latent_dim=LATENT_DIM,
):
    """Load a trained Koopman model from checkpoint"""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Get normalization stats
    norm_stats = checkpoint["normalization_stats"]

    # Create model (need to know dimensions)
    if state_dim is None or action_dim is None:
        raise ValueError("Must provide state_dim and action_dim to load model")

    model = KoopmanOperator(state_dim, action_dim, latent_dim)
    model.set_normalization_params(
        norm_stats["state_mean"],
        norm_stats["state_std"],
        norm_stats["action_mean"],
        norm_stats["action_std"],
    )

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Loaded model from epoch {checkpoint['epoch']} with validation loss: {checkpoint['val_loss']:.6f}"
    )

    return model, norm_stats


def plot_predictions_vs_actual(
    predictions, actual_states, states_tensor, sample_idx=0, max_steps=None
):
    """Plot predicted vs actual state trajectories as line charts"""

    # Convert tensors to numpy
    pred_np = predictions[0].cpu().numpy()  # [T-1, state_dim]
    actual_np = actual_states[0].cpu().numpy()  # [T-1, state_dim]
    initial_state = states_tensor[0, 0].cpu().numpy()  # [state_dim]

    state_dim = pred_np.shape[1]
    if max_steps is None:
        max_steps = pred_np.shape[0]
    else:
        max_steps = min(max_steps, pred_np.shape[0])

    # Create time steps (starting from 1 since we predict next states)
    time_steps = np.arange(1, max_steps + 1)

    # Create subplots - one for each state dimension
    fig, axes = plt.subplots(state_dim, 1, figsize=(12, 2 * state_dim))
    if state_dim == 1:
        axes = [axes]

    fig.suptitle(
        f"Sample {sample_idx + 1}: Predicted vs Actual State Trajectories",
        fontsize=14,
        fontweight="bold",
    )

    for state_idx in range(state_dim):
        ax = axes[state_idx]

        # Plot initial state as a point
        ax.plot(
            0,
            initial_state[state_idx],
            "go",
            markersize=8,
            label="Initial State",
            zorder=5,
        )

        # Plot predicted and actual trajectories
        ax.plot(
            time_steps,
            pred_np[:max_steps, state_idx],
            "r-",
            linewidth=2,
            label="Predicted",
            marker="o",
            markersize=4,
        )
        ax.plot(
            time_steps,
            actual_np[:max_steps, state_idx],
            "b-",
            linewidth=2,
            label="Actual",
            marker="s",
            markersize=4,
        )

        # Formatting
        ax.set_xlabel("Time Step")
        ax.set_ylabel(f"State Dimension {state_idx + 1}")
        ax.set_title(f"State Dimension {state_idx + 1}")
        ax.grid(True, alpha=0.3)
        # ax.set_ylim(-1, 1)

        if state_idx == 0:
            ax.legend()

        # Add error statistics as text
        errors = np.abs(
            pred_np[:max_steps, state_idx] - actual_np[:max_steps, state_idx]
        )
        mae = np.mean(errors)
        mse = np.mean(errors**2)
        ax.text(
            0.02,
            0.98,
            f"MAE: {mae:.4f}\nMSE: {mse:.4f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()
    return fig


def generate_sample_prediction_with_plots(
    model, env, seq_len=SEQ_LENGTH, num_samples=1, save_plots=True
):
    """Generate sample inputs and show model predictions with visualizations"""
    model.eval()
    device = next(model.parameters()).device
    model.to(device)

    print("=" * 80)
    print("SAMPLE PREDICTION DEMONSTRATION WITH PLOTS")
    print("=" * 80)

    all_figures = []

    with torch.no_grad():
        for sample_idx in range(num_samples):
            print(f"\n--- Sample {sample_idx + 1} ---")

            # Reset environment and collect initial sequence
            state = env.reset(seed=122342)
            if isinstance(state, tuple):
                state = state[0]

            states = [state.copy()]
            actions = []

            print(f"Initial state: {state}")

            # Collect sequence of states and actions
            for step in range(seq_len - 1):
                # Take random action
                action = env.action_space.sample()
                next_state, _, done, truncated, _ = env.step(action)
                if isinstance(next_state, tuple):
                    next_state = next_state[0]

                actions.append(action.copy() if hasattr(action, "copy") else action)
                states.append(next_state.copy())

                print(f"Step {step + 1}: Action = {action}")

                if done or truncated:
                    print("Episode terminated early")
                    break

            if len(states) < 2:
                print("Not enough states collected, skipping...")
                continue

            # Prepare tensors for model
            states_tensor = torch.FloatTensor(states).unsqueeze(0).to(device)
            actions_tensor = torch.FloatTensor(actions).unsqueeze(0).to(device)

            print(
                f"Input shapes - States: {states_tensor.shape}, Actions: {actions_tensor.shape}"
            )

            # Get model predictions
            predictions, koopman_states = model(states_tensor, actions_tensor)
            actual_next_states = states_tensor[:, 1:, :]

            print(
                f"Output shapes - Predictions: {predictions.shape}, Koopman states: {koopman_states.shape}"
            )

            # Calculate and print error statistics
            all_errors = torch.abs(predictions - actual_next_states)
            mean_error = torch.mean(all_errors).item()
            max_error = torch.max(all_errors).item()
            mse = torch.mean((predictions - actual_next_states) ** 2).item()

            print(f"Sample {sample_idx + 1} Statistics:")
            print(f"  Mean Absolute Error: {mean_error:.6f}")
            print(f"  Max Absolute Error:  {max_error:.6f}")
            print(f"  Mean Squared Error:  {mse:.6f}")

            # Create and show plot
            fig = plot_predictions_vs_actual(
                predictions,
                actual_next_states,
                states_tensor,
                sample_idx,
                max_steps=min(SEQ_LENGTH, predictions.shape[1]),
            )
            all_figures.append(fig)

            if save_plots:
                plot_filename = f"prediction_sample_{sample_idx + 1}.png"
                fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
                print(f"  Plot saved as: {plot_filename}")

            plt.show()
            print("\n" + "=" * 60)

    return all_figures


def demonstrate_model_capabilities_with_plots(
    checkpoint_path="best_koopman_model.pth", env_name="Pendulum-v1"
):
    """Complete demonstration of loading and using the trained model with visualizations"""

    # Create environment to get dimensions
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]

    if hasattr(env.action_space, "n"):  # Discrete
        action_dim = env.action_space.n
    else:  # Continuous
        action_dim = env.action_space.shape[0]

    print(f"Loading model for environment: {env_name}")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

    try:
        # Load the trained model
        model, norm_stats = load_model(checkpoint_path, state_dim, action_dim, 24)

        print(f"\nNormalization statistics:")
        print(f"State mean: {norm_stats['state_mean']}")
        print(f"State std:  {norm_stats['state_std']}")
        print(f"Action mean: {norm_stats['action_mean']}")
        print(f"Action std:  {norm_stats['action_std']}")

        # Generate sample predictions with plots
        figures = generate_sample_prediction_with_plots(
            model, env, seq_len=15, num_samples=1, save_plots=True
        )

        # Quick evaluation
        print("\n" + "=" * 80)
        print("QUICK EVALUATION")
        print("=" * 80)
        mean_error, std_error = evaluate_model(model, env, num_episodes=5)

        return figures

    except FileNotFoundError:
        print(f"Model checkpoint not found at {checkpoint_path}")
        print("Please train a model first using the main training function.")
        return []
    except Exception as e:
        print(f"Error loading or using model: {e}")
        return []
    finally:
        env.close()


def generate_sample_prediction_with_plots_minari(
    model: KoopmanOperator,
    dataset: minari.MinariDataset,
    means,
    stds,
    seq_len=SEQ_LENGTH,
    num_samples=1,
    save_plots=True,
):
    """Generate sample inputs and show model predictions with visualizations"""
    model.eval()
    device = next(model.parameters()).device
    model.to(device)

    print("=" * 80)
    print("SAMPLE PREDICTION DEMONSTRATION WITH PLOTS")
    print("=" * 80)

    all_figures = []

    with torch.no_grad():
        for sample_idx in range(num_samples):
            print(f"\n--- Sample {sample_idx + 1} ---")

            episode = dataset.sample_episodes(1)[0]
            state = episode.observations[0]

            states = [state]
            actions = []

            print(f"Initial state: {state}")

            # Collect sequence of states and actions
            for step in range(seq_len - 1):
                # Take random action
                action = episode.actions[step]
                next_state = episode.observations[step]

                actions.append(action)
                states.append(next_state)

            # Prepare tensors for model
            states_tensor = torch.FloatTensor(states).unsqueeze(0).to(device)
            actions_tensor = torch.FloatTensor(actions).unsqueeze(0).to(device)

            print(
                f"Input shapes - States: {states_tensor.shape}, Actions: {actions_tensor.shape}"
            )

            # Get model predictions
            predictions = model.multi_step_prediction(
                states_tensor[:, 0, :], actions_tensor
            )
            # predictions, latent_states = model(states_tensor, actions_tensor)

            actual_next_states = states_tensor[:, 1:, :]

            print(f"Output shapes - Predictions: {predictions.shape}")

            # Calculate and print error statistics
            all_errors = torch.abs(predictions - actual_next_states)
            mean_error = torch.mean(all_errors).item()
            max_error = torch.max(all_errors).item()
            mse = torch.mean((predictions - actual_next_states) ** 2).item()

            print(f"Sample {sample_idx + 1} Statistics:")
            print(f"  Mean Absolute Error: {mean_error:.6f}")
            print(f"  Max Absolute Error:  {max_error:.6f}")
            print(f"  Mean Squared Error:  {mse:.6f}")

            print(predictions.shape, means.shape, stds.shape)
            # Unnormalize predictions and actual_next_states for plotting
            means = torch.tensor(
                means, dtype=predictions.dtype, device=predictions.device
            )
            stds = torch.tensor(
                stds, dtype=predictions.dtype, device=predictions.device
            )

            # Reshape for broadcasting: (1, 1, state_dim)
            means = means.view(1, 1, -1)
            stds = stds.view(1, 1, -1)

            predictions = predictions * stds + means
            actual_next_states = actual_next_states * stds + means
            states_tensor = states_tensor * stds + means

            # Create and show plot
            fig = plot_predictions_vs_actual(
                predictions,
                actual_next_states,
                states_tensor,
                sample_idx,
                max_steps=min(SEQ_LENGTH, predictions.shape[1]),
            )
            all_figures.append(fig)

            if save_plots:
                plot_filename = f"prediction_sample_{sample_idx + 1}.png"
                fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
                print(f"  Plot saved as: {plot_filename}")

            plt.show()
            print("\n" + "=" * 60)

    return all_figures


def demonstrate_model_capabilities_with_plots_minari(
    checkpoint_path="best_koopman_model.pth",
    dataset_name="mujoco/inverteddoublependulum/expert-v0",
):
    """Complete demonstration of loading and using the trained model with visualizations"""

    dataset = minari.load_dataset(dataset_name, download=True)
    env = dataset.recover_environment()

    # Create environment to get dimensions
    state_dim = env.observation_space.shape[0]

    if hasattr(env.action_space, "n"):  # Discrete
        action_dim = env.action_space.n
    else:  # Continuous
        action_dim = env.action_space.shape[0]

    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

    try:
        # Load the trained model
        model, norm_stats = load_model(checkpoint_path, state_dim, action_dim, 24)

        print(f"\nNormalization statistics:")
        print(f"State mean: {norm_stats['state_mean']}")
        print(f"State std:  {norm_stats['state_std']}")
        print(f"Action mean: {norm_stats['action_mean']}")
        print(f"Action std:  {norm_stats['action_std']}")

        # Generate sample predictions with plots
        figures = generate_sample_prediction_with_plots_minari(
            model,
            dataset,
            norm_stats["state_mean"],
            norm_stats["state_std"],
            seq_len=20,
            num_samples=1,
            save_plots=True,
        )

        # Quick evaluation
        print("\n" + "=" * 80)
        print("QUICK EVALUATION")
        print("=" * 80)
        mean_error, std_error = evaluate_model(model, env, num_episodes=5)

        return figures

    except FileNotFoundError:
        print(f"Model checkpoint not found at {checkpoint_path}")
        print("Please train a model first using the main training function.")
        return []
    except Exception as e:
        print(f"Error loading or using model: {e}")
        return []
    finally:
        env.close()


# Example usage
if __name__ == "__main__":
    # Train the model
    # model = train_using_random_sampling(
    #     env_name=ENV_NAME,
    #     num_trajectories=500,
    #     max_traj_len=20,
    #     seq_len=SEQ_LENGTH,
    #     latent_dim=LATENT_DIM,
    #     num_epochs=300,
    #     batch_size=32,
    #     learning_rate=1e-3,
    # )
    model = train_using_expert_dataset(
        dataset_name=DATASET_NAME,
        num_trajectories=500,
        max_traj_len=20,
        seq_len=SEQ_LENGTH,
        latent_dim=LATENT_DIM,
        num_epochs=300,
        batch_size=32,
        learning_rate=1e-3,
    )

    # Demonstrate model loading and sample generation with plots
    print("\n" + "=" * 80)
    print("DEMONSTRATING MODEL LOADING AND SAMPLE PREDICTION WITH PLOTS")
    print("=" * 80)
    # figures = demonstrate_model_capabilities_with_plots(
    #     "final_koopman_model.pth", ENV_NAME
    # )
    figures = demonstrate_model_capabilities_with_plots_minari(
        "final_koopman_model.pth",
        DATASET_NAME,
    )
