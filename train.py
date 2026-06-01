import logging
import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from ml_foundations.config import MLPConfig, TrainingConfig
from ml_foundations.models.network import MLP

#LOGGING SETUP
 
logging.basicConfig(
     level=logging.INFO,
     format="%(asctime)s | %(levelname)s | %(message)s",
     datefmt="%H:%M:%S",
 )
logger = logging.getLogger(__name__)

#DATA GENERATION

def make_sine_dataset(
        n_samples: int = 1000,
        noise: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a noisy sine wave regression dataset.
    
    Returns:
        X: shape (n_samples, 1) - input values in [-pi, pi]
        y: shape (n_samples, 1) - noisy sine values
    """
    x = np.linspace(-np.pi, np.pi, n_samples).astype(np.float32)
    y = np.sin(x) + np.random.randn(n_samples).astype(np.float32) * noise

    #Reshape to (n_samples, 1) - the model expects 2D input
    X = torch.from_numpy(x.reshape(-1, 1))
    Y = torch.from_numpy(y.reshape(-1, 1))
    return X , Y 

def make_dataloaders(
    X: torch.Tensor, 
    Y: torch.Tensor,
    val_split: float,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Split data into train/val and wrap in dataLoaders."""
    n_val = int(len(X) * val_split)
    n_train = len(X) - n_val
    #Shuffle before splitting
    indices = torch.randperm(len(X))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_ds = TensorDataset(X[train_idx], Y[train_idx])
    val_ds = TensorDataset(X[val_idx], Y[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

#TRAINING LOOP
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
) -> dict[str, list[float]]:
    """
    Full training loop with validation and checkpointing.
    Returns loss history for plotting.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    #Create checkpoint directory if needed
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[float]] = {"train": [], "val": []}

    for epoch in range(config.epochs):

        #TRAIN
        model.train()
        train_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        #VALIDATE
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(config.device)
                batch_y = batch_y.to(config.device)
                preds = model(batch_x)
                val_losses.append(criterion(preds, batch_y).item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        #LOG
        if epoch % config.log_interval == 0:
            logger.info(
                f"Epoch {epoch:4d}/{config.epochs} | "
                f"Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f}"
            )

        #CHECKPOINT
        if epoch % 50 == 0:
            ckpt_path = config.checkpoint_dir / f"ckpt_epoch_{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, ckpt_path)

    return history

#PLOTTING
def plot_results(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    history: dict[str, list[float]],
) -> None:
    """Two-panel plot: loss curves +model predictions vs ground truth."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    #Panel 1: Loss curves
    ax1.plot(history["train"], label="Train Loss")
    ax1.plot(history["val"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training Curves")
    ax1.legend()
    ax1.set_yscale("log")

    #Panel 2: Predictions vs Ground Truth
    model.eval()
    with torch.no_grad():
        preds = model(X).numpy()

    x_np = X.numpy().flatten()
    y_np = Y.numpy().flatten()
    p_np = preds.flatten()

    sort_idx = np.argsort(x_np) #sort by x for clean line plot
    ax2.scatter(x_np, y_np, s=4, alpha=0.3, label="Data (noisy)")
    ax2.plot(x_np[sort_idx], p_np[sort_idx], color="red", 
             linewidth=2, label="Model prediction")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("Sine Wave Approximation")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("experiments/training_results.png", dpi=150)
    logger.info("Plot saved to experiments/training_results.png")
    plt.show()

#ENTRY POINT
if __name__ == "__main__":
    torch.manual_seed(42) #for reproducibility
    np.random.seed(42)

    #Configuration - all decisions live here, nothing hardcoded below
    arch_config = MLPConfig(
        input_dim = 1,
        hidden_dims = [64, 64, 64],
        output_dim = 1,
        activation = "relu",
        dropout = 0.0,
    )

    train_config = TrainingConfig(
        epochs = 200,
        learning_rate = 1e-3,
        batch_size = 32,
        device = "cpu",
        log_interval = 20,
    )

    #Build model
    model = MLP(arch_config).to(train_config.device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model built | Parameters: {n_params:,}")
    logger.info(f"Architecture:\n{model}")

    #Data
    X, Y = make_sine_dataset(n_samples=1000, noise=0.05)
    train_loader, val_loader = make_dataloaders(
        X , Y, 
        val_split = train_config.val_split,
        batch_size = train_config.batch_size,
    )
    logger.info(f"Dataset ready | {len(X)} samples")

    #Train
    history = train(model, train_loader, val_loader, train_config)

    #Results
    final_val = history["val"][-1]
    logger.info(f"Training complete | Final Val Loss: {final_val:.5f}")

    plot_results(model, X, Y, history)
     
