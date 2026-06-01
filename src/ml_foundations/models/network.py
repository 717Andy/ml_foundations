import torch
import torch.nn as nn
from ml_foundations.config import MLPConfig

class MLP(nn.Module):
    """
    A configurable Multi-Layer Perceptron.

    Architecture: Linear -> Activation -> Droupout -> (repeated) -> Linear (output)
    The final layer has NO activation - raw logits/values for regression or
    classification head to handle.

    Args:
        config: MLPConfig dataclass controllingall architectural choices.

    Example:
        >>> config = MLPConfig(input_dim=10, hidden_dim=[64,64], output_dim=1])
        >>> model = MLPConfig(config)
        >>> x torch.rand(32,10) #batch of 32 samples
        >>> out = model(x) #shape (32,1)
    """

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config
        self.network = self._build_network()

    def _build_network(self) -> nn.Sequential:
        """
        Dynamically builds layer stack fron config.
        Pattern: [Linear -> Activation -> Dropout] x N -> Linear (output, no activation)
        """
        #All dimensions in order: input -> hidden... -> output
        dims = [self.config.input_dim] + self.config.hidden_dims + [self.config.output_dim]

        #Pick the activation function object(not called yet, just selected)
        if self.config.activation == "relu":
            activation_cls = nn.ReLU
        elif self.config.activation == "tanh":
            activation_cls = nn.Tanh
        else:
            raise ValueError(
                f"Unknown activation '{self.config.activation}'."
                f"Choose 'relu' or 'tanh'."
            )
        
        layers: list[nn.Module] = []

        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            layers.append(nn.Linear(in_dim, out_dim))

            is_last_layer = (i == len(dims) - 2) 

            if not is_last_layer:
                layers.append(activation_cls())

                if self.config.dropout > 0.0:
                    layers.append(nn.Dropout(p=self.config.dropout))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)
    