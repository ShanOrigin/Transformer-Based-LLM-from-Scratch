import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-Wise Feed-Forward Network (MLP) for Transformer Blocks.

    Applies two linear transformations with a GELU non-linear activation in
    between to each token position independently:
        FFN(x) = Linear_2(GELU(Linear_1(x)))

    Expands feature dimensionality by `expansion_factor` (default 4x) in the
    hidden layer and projects back down to `d_model`.

    Attributes:
        d_model (int): Input and output feature dimensionality.
        hidden_dim (int): Expanded intermediate dimensionality (d_model * expansion_factor).
    """

    def __init__(
        self,
        d_model: int,
        expansion_factor: int = 4,
        init_std: float = 0.02,
    ) -> None:
        """
        Initializes the two-layer position-wise feed-forward network.

        Args:
            d_model (int): Model embedding dimension (e.g., 128).
            expansion_factor (int): Multiplier for hidden layer width. Defaults to 4.
            init_std (float): Standard deviation for Gaussian weight initialization.

        Raises:
            ValueError: If d_model <= 0, expansion_factor <= 0, or init_std <= 0.
        """
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if expansion_factor <= 0:
            raise ValueError(
                f"expansion_factor must be positive, got {expansion_factor}."
            )
        if init_std <= 0:
            raise ValueError(f"init_std must be positive, got {init_std}.")

        self.d_model: int = d_model
        self.hidden_dim: int = d_model * expansion_factor

        # Layer 1: Expands (batch_size, seq_len, d_model) -> (batch_size, seq_len, hidden_dim)
        self.fc1: nn.Linear = nn.Linear(d_model, self.hidden_dim)

        # Smooth non-linear activation standard in modern GPT architectures
        self.activation: nn.GELU = nn.GELU()

        # Layer 2: Compresses (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, d_model)
        self.fc2: nn.Linear = nn.Linear(self.hidden_dim, d_model)

        self._init_weights(std=init_std)

    def _init_weights(self, std: float) -> None:
        """Initializes linear weights with N(0, std^2) and biases to zero."""
        nn.init.normal_(self.fc1.weight, mean=0.0, std=std)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=std)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the position-wise feed-forward transformation to each token.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).

        Raises:
            TypeError: If x is not a floating-point tensor.
            ValueError: If x is not 3D or its feature dimension != d_model.
        """
        if not torch.is_floating_point(x):
            raise TypeError(f"Expected floating-point tensor, got {x.dtype}.")
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input (batch_size, seq_len, d_model), got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Input dimension ({x.shape[-1]}) does not match d_model ({self.d_model})."
            )

        # Step 1: Expand to hidden_dim
        hidden: torch.Tensor = self.fc1(x)

        # Step 2: Apply non-linear activation
        activated: torch.Tensor = self.activation(hidden)

        # Step 3: Project back to d_model
        return self.fc2(activated)

    @property
    def num_parameters(self) -> int:
        """Returns total learnable parameters in the feed-forward network."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, hidden_dim={self.hidden_dim}"