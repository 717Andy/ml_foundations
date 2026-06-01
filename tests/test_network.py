import torch
import pytest
from ml_foundations.models.network import MLP
from ml_foundations.config import MLPConfig

class TestMLPShapes:
    """Tests that the model produces correct output shapes"""
    def test_basic_output_shape(self):
        """Output shaoe must be (bastch_size, output_dim)"""
        config = MLPConfig(input_dim=10, hidden_dims=[64, 32],output_dim=5)
        model = MLP(config)
        x = torch.randn(16, 10)
        out = model(x)
        assert out.shape == (16, 5), f"Expected (16, 5), got {out.shape}"

    def test_batch_size_one(self):
        """Model must handle a single sample (batch_size=1)"""
        config = MLPConfig(input_dim=4, hidden_dims=[16], output_dim=2)
        model = MLP(config)
        x = torch.randn(1, 4)
        out = model(x)
        assert out.shape == (1, 2)

    def test_no_hidden_layers(self):
        """An MLP with no hidden layers is just a linear transform"""
        config = MLPConfig(input_dim=8, hidden_dims=[], output_dim=3)
        model = MLP(config)
        x = torch.randn(5, 8)
        out = model(x)
        assert out.shape == (5, 3)

    def test_deep_networks(self):
        """Deep networks (5 hidden layers) should still produce correct shapes"""
        config = MLPConfig(
            input_dim=10,
            hidden_dims=[128, 64, 64, 32, 16],
            output_dim=1
        )
        model = MLP(config)
        x = torch.randn(8, 10)
        out = model(x)
        assert out.shape == (8, 1)

class TestMLPGradients:
    """Test that gradients flow correctly through the network"""

    def test_all_params_receive_gradients(self):
        """Every parameter must have a non-None gradient after a backward pass.
        If any gradient is None, that parameter is disconnected from the loss and won't learn"""
        config = MLPConfig(input_dim=4, hidden_dims=[16, 8], output_dim=2)
        model = MLP(config)
        x = torch.randn(4, 4)
        loss = model(x).sum() #simplest possible loss: sum of all outputs
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient."
                f"It is disconnected from the computation graph."
            )

    def test_gradients_are_nonzero(self):
        """Gradients should not be all zeros for a random input"""
        config = MLPConfig(input_dim=4, hidden_dims=[16], output_dim=1)
        model = MLP(config)
        x = torch.randn(4, 4)
        loss = model(x).sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad.abs().sum() > 0, (
                f"Parameter '{name}' has all-zero gradients - "
                f"suspect dead neurons or disconnected layers."
            )

class TestMLPBehavior:
    """Test that the model behaves correctly in train vs eval modes"""

    def test_eval_mode_deterministic(self):
        """In eval mode with dropout, same input must give same output.
        (Dropout is random during training but disabled during eval)"""
        config = MLPConfig(input_dim=10, hidden_dims=[64], output_dim=4, dropout=0.5)
        model = MLP(config).eval()    # <-- eval mode
        x = torch.randn(8, 10)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1, out2), (
            "Model is non-deterministic in eval mode."
            "Check that dropout is properly disabled."
        )

    def test_train_mode_stochastic_with_dropouts(self):
        """In train mode with high dropout, same input should give different outputs.
        (This verifies dropout is actually active during training)"""
        config = MLPConfig(input_dim=10, hidden_dims=[64], output_dim=4, dropout=0.9)
        model = MLP(config).train()   # <-- train mode
        x = torch.randn(8, 10)
        out1 = model(x)
        out2 = model(x)

        #With 90% dropout, outputs should almost certainly differ
        assert not torch.allclose(out1, out2), (
            "Outputs are identical in train modewith 0.9 dropout - "
            "dropout may not be active."
        )

    def test_invalid_activation_raises(self):
        """Passing an unknown activation should raise ValueError, not silently fail"""
        config = MLPConfig(input_dim=4, hidden_dims=[16], output_dim=2, activation="sigmoid")
        with pytest.raises(ValueError, match="Unknown activation"):
            MLP(config)

class TestMLPLearning:
    """Tests that the model can actually learn something"""

    def test_loss_decreases_in_trivial_task(self):
        """A model should be able to overfit a tiny dataset.
        If loss doesn't decrease, something is fundamentally broken"""
        torch.manual_seed(0)

        config = MLPConfig(input_dim=1, hidden_dims=[32, 32], output_dim=1)
        model = MLP(config)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = torch.nn.MSELoss()

        #10 samples of y = x^2
        x = torch.linspace(-1, 1, 10).unsqueeze(1)
        y = x ** 2

        #Record initial loss
        model.eval()
        with torch.no_grad():
            initial_loss = loss_fn(model(x), y).item()

        #Train for 300 steps
        model.train()
        for _ in range(300):
            optim.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optim.step()

        #Record final loss
        model.eval()
        with torch.no_grad():
            final_loss = loss_fn(model(x), y).item()

        assert final_loss < initial_loss * 0.1, (
            f"Loss did not decrease sufficiently. "
            f"Initial: {initial_loss:.4f}, Final: {final_loss:.4f}. "
            f"Expected final < {initial_loss * 0.1:.4f}."
        )

