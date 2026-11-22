import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Learnable Absolute Positional Embedding Layer for injecting token order
    information into token embedding sequences.

    Generates positional vectors for indices [0, seq_len - 1] from a learnable
    lookup table of shape (context_size, d_model) and adds them directly to
    the input token embeddings via broadcasting.

    Attributes:
        context_size (int): Maximum sequence length supported by the model.
        d_model (int): Dimensionality of each positional embedding vector.
        table (nn.Embedding): Underlying learnable lookup matrix of shape
                              (context_size, d_model).
    """

    def __init__(
        self,
        context_size: int,
        d_model: int,
        init_std: float = 0.02,
    ) -> None:
        """
        Initializes the positional embedding lookup table with Gaussian weights.

        Args:
            context_size (int): Maximum sequence length (block_size, e.g., 128).
            d_model (int): Embedding vector dimension per position (e.g., 128).
            init_std (float): Standard deviation for normal weight initialization.
                              Defaults to 0.02 (standard GPT initialization).

        Raises:
            ValueError: If context_size <= 0, d_model <= 0, or init_std <= 0.
        """
        super().__init__()

        if context_size <= 0:
            raise ValueError(f"context_size must be positive, got {context_size}.")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if init_std <= 0:
            raise ValueError(f"init_std must be positive, got {init_std}.")

        self.context_size: int = context_size
        self.d_model: int = d_model

        # Create the learnable positional lookup matrix: shape (context_size, d_model)
        self.table: nn.Embedding = nn.Embedding(
            num_embeddings=self.context_size,
            embedding_dim=self.d_model,
        )

        self._init_weights(std=init_std)

    def _init_weights(self, std: float) -> None:
        """
        Initializes positional embedding weights from a zero-mean normal distribution
        N(0, std^2) for stable gradient flow during early training.

        Args:
            std (float): Standard deviation of the normal distribution.
        """
        nn.init.normal_(self.table.weight, mean=0.0, std=std)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Adds learned positional embeddings to a batch of token embedding sequences.

        Args:
            token_embeddings (torch.Tensor): 3D floating-point tensor of token
                                             embeddings with shape
                                             (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Position-aware embeddings with identical shape
                          (batch_size, seq_len, d_model).

        Raises:
            TypeError: If token_embeddings is not a floating-point tensor.
            ValueError: If token_embeddings is not 3D, its last dimension does
                        not match d_model, or seq_len exceeds context_size.
        """
        if not torch.is_floating_point(token_embeddings):
            raise TypeError(
                f"token_embeddings must be a floating-point tensor, got {token_embeddings.dtype}."
            )

        if token_embeddings.ndim != 3:
            raise ValueError(
                f"Expected 3D input tensor (batch_size, seq_len, d_model), "
                f"got shape {tuple(token_embeddings.shape)}."
            )

        _, seq_len, emb_dim = token_embeddings.shape

        if emb_dim != self.d_model:
            raise ValueError(
                f"Input embedding dimension ({emb_dim}) does not match "
                f"PositionalEmbedding d_model ({self.d_model})."
            )

        if seq_len > self.context_size:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds maximum context_size ({self.context_size})."
            )

        # Create position indices [0, 1, ..., seq_len - 1] on the input tensor's device
        positions: torch.Tensor = torch.arange(
            start=0,
            end=seq_len,
            dtype=torch.long,
            device=token_embeddings.device,
        )

        # Lookup positional vectors: shape (seq_len, d_model)
        pos_embeddings: torch.Tensor = self.table(positions)

        # Broadcast across batch_size and add: (B, T, d_model) + (T, d_model) -> (B, T, d_model)
        return token_embeddings + pos_embeddings

    @property
    def num_parameters(self) -> int:
        """Returns the total number of learnable parameters in the positional table."""
        return self.context_size * self.d_model

    def extra_repr(self) -> str:
        """Provides a clean summary string when printing the module."""
        return f"context_size={self.context_size}, d_model={self.d_model}"

