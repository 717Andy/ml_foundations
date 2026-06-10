"""
Neural network implemented in pure NumPy.
No autograd. Every gradient computed by hand.
Goal: understand exactly what PyTorch's backward() does
"""

import numpy as np
import torch
import torch.nn as nn


class Linear:
    """Fully connected linear layer: output = X @ W + b"""

    def __init__(self, in_features: int, out_features: int):
        #The initialization: scale by sqrt(2 / fan_in)
        #Keeps gradient magnitudes stable through deep networks
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)

        #Gradients - populated during backward()
        self.grad_W: np.ndarray | None = None
        self.grad_b: np.ndarray | None = None
        #Cache input for use in backward pass 
        self._X: np.ndarray | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        X: (batch, in_features)
        Return: (batch, out_features)
        """
        self._X = X     #Cache for backward
        return X @ self.W + self.b
    
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        grad_out: (batch, out_features)
            gradient of loss w.r.t. this layer's output
        
        Computes:
            self.grad_W - gradient w.r.t. weights
            self.grad_b - gradient w.r.t. bias
            returns grad_X - gradient w.r.t. input (passed to previous layer)
        """
        self.grad_W = self._X.T @ grad_out
        self.grad_b = np.sum(grad_out, axis=0)
        grad_X = grad_out @ self.W.T
        return grad_X
    def step(self, lr: float) -> None:
        """Gradient descent update for this layer's parameters"""
        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b


class ReLU:
    """ReLU activation: output = max(0, x)"""

    def __init__(self):
        self._X: np.ndarray | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._X = X
        return np.maximum(0, X)
    
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        Gradient of Relu: passes grad_out where input > 0, blocks it elsewhere
        """
        return grad_out * (self._X > 0).astype(np.float32)
    
    def step(self, lr: float) -> None:
        pass             # ReLU has no parameters, nothing to update 


class MSELoss:
    """Mean Squaered Error loss: mean((predictions - targets)^2)"""

    def __init__(self):
        self._predictions: np.ndarray | None = None
        self._targets: np.ndarray | None = None

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        self._predictions = predictions
        self._targets = targets
        return np.mean((predictions - targets) ** 2)
    
    def backward(self) -> np.ndarray:
        """
        Gradient of MSE w.r.t. predictions.
        This is where the gradient signal originates.
        Shape matches predictions.
        """
        N = self._predictions.shape[0]
        return 2 * (self._predictions - self._targets) / N
    

class NumpyMLP:
    """
    Multi-layer perceptron using our hand-built layers.
    Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear (output)
    """

    def __init__(self, layer_dims: list[int]):
        """
        layer_dims: [input_dim, hidden1, hidden2, ...., output_dim]
        e.g. [1, 64, 64, 1] builds:
        Linear(1->64) -> ReLU -> Linear(64->64) -> ReLU -> Linear(64->1)
        """
        self.layers: list[Linear| ReLU] = []

        for i in range(len(layer_dims) - 1):
            self.layers.append(Linear(layer_dims[i], layer_dims[i + 1]))

            #Add ReLU after every layer except the last
            if i < len(layer_dims) - 2:
                self.layers.append(ReLU())

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Pass X through every layer in order"""
        out = X 
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self, grad: np.ndarray) -> None:
        """Pass gradient backward through every layer in reverse order"""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, lr: float) -> None:
        """Update parameters in every layer"""
        for layer in self.layers:
            layer.step(lr)


def train_numpy_mlp(
        layer_dims: list[int] = [1, 64, 64, 1],
        lr: float = 1e-2,
        epochs: int = 1000,
        n_samples: int = 500,
) -> list[float]:
    """
    Full Training loop. Returns loss history
    """
    np.random.seed(42)

    #Data: noisy sine wave
    x = np.linspace(-np.pi, np.pi, n_samples).astype(np.float32).reshape(-1, 1)
    x = (x - x.mean()) / x.std()   #Normalize input to zero mean, unit variance
    y = (np.sin(x) + np.random.randn(*x.shape).astype(np.float32) * 0.05)

    model = NumpyMLP(layer_dims)
    loss_fn = MSELoss()
    history = []

    for epoch in range(epochs):
        #Forward pass
        predictions = model.forward(x)
        #Compute loss
        loss = loss_fn.forward(predictions, y)
        history.append(loss)

        #Backward pass - get gradients from loss, propagate back
        grad = loss_fn.backward()
        model.backward(grad)

        #Update weights
        model.step(lr)

        if epoch % 100 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

    return history

if __name__ == "__main__":
    history = train_numpy_mlp()
    final_loss = history[-1]
    print(f"Final_loss: {final_loss:.6f}")

    if final_loss < 0.01:
        print("PASSED - loss converged below 0.01")
    else:
        print("FAILED - loss did not converge. Debug the backward passes")


#Build equivalent PyTorch model with same init
torch.manual_seed(0)
np_model = NumpyMLP([1, 8, 1]) #small for easy comparison

pt_model = nn.Sequential(
    nn.Linear(1, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

#Copy numpy weight INTO pytorch model
with torch.no_grad():
    linear_layers = [l for l in np_model.layers if isinstance(l, Linear)]
    linear_idx = 0
    for pt_layer in pt_model:
        if isinstance(pt_layer, nn.Linear):
            pt_layer.weight.copy_(torch.from_numpy(linear_layers[linear_idx].W.T))  #Note PyTorch's weight shape is (out_features, in_features)
            pt_layer.bias.copy_(torch.from_numpy(linear_layers[linear_idx].b))
            linear_idx += 1

#Run one forward + backward through both
x_np = np.random.randn(4, 1).astype(np.float32)
x_pt = torch.from_numpy(x_np).requires_grad_(False)

#NumPy forward + backward
y_np = np_model.forward(x_np)
loss_fn = MSELoss()
target = np.zeros_like(y_np)
loss_fn.forward(y_np, target)
grad = loss_fn.backward()
np_model.backward(grad)
np_grad_W = [l for l in np_model.layers if isinstance(l, Linear)][0].grad_W

#PyTorch forward + backward
y_pt = pt_model(x_pt)
pt_loss_fn = torch.nn.MSELoss()
loss_pt = pt_loss_fn(y_pt, torch.zeros_like(y_pt)) #Target is zero to match NumPy loss
loss_pt.backward()
pt_grad_W = pt_model[0].weight.grad.numpy().T  #Transpose back to match NumPy shape

#Compare gradients
match = np.allclose(np_grad_W, pt_grad_W, atol=1e-5)
print(f"\nGradient match: {'YES' if match else 'NO'}")
print(f"NumPy grad_W[:3]: {np_grad_W.flatten()[:3]}")
print(f"PyTorch grad_W[:3]: {pt_grad_W.flatten()[:3]}")
