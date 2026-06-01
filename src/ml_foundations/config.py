from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class MLPConfig:
    """"
    Configuration for the MLP architecture .
    All architecture decisions live here - nothing hardcoded in the model
    """
    input_dim: int 
    hidden_dims: list[int]
    output_dim: int 
    activation: str = "relu"
    dropout: float = 0.0

@dataclass
class TrainingConfig:
    """
    Configuration for the training loop
    """
    epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 32
    device: str ="cpu"
    log_interval: int = 10
    checkpoint_dir: Path = field(
        default_factory=lambda: Path("experiments/checkpoints")
    )
    val_split: float = 0.2

