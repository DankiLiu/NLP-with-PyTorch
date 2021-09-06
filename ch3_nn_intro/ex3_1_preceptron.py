import torch
import torch.nn as nn


class Preceptron(nn.Module):
    """A perceptron is a linear layer."""
    def __init__(self, input_dim):
        """
        Args:
            input_dim (int): size of input.
        """
        super(Preceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x_in):
        """Forward pass of the preceptron.
        Args:
            x_in (torch.Tensor): a input tensor
                x_in has shape of (batch, num_features).
        Returns:
            output tensor, should has shape (batch,)
        """
        return torch.sigmoid(self.fc1(x_in)).squeeze()
