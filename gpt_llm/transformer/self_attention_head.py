import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    """
    Single Head of Causal (Masked) Scaled Dot-Product Self-Attention.

    Projects input embeddings of shape (batch_size, seq_len, d_model) into
    Query (Q), Key (K), and Value (V) representations of dimension `head_dim`,
    computes scaled affinity scores, applies a lower-triangular causal mask so
    tokens cannot attend to future positions, and aggregates Value vectors.

    Attributes:
        d_model (int): Input embedding dimensionality.
        head_dim (int): Dimensionality of Q, K, and V projections for this head.
        context_size (int): Maximum sequence length supported by the causal mask.
        scale (float): Scaling factor 1.0 / sqrt(head_dim) for dot-product stability.
    """

    def __init__(
        self,
        d_model: int,
        head_dim: int,
        context_size: int,
        init_std: float = 0.02,
    ) -> None:
        """
        Initializes Query, Key, and Value linear projections and registers the
        lower-triangular causal mask buffer.

        Args:
            d_model (int): Model embedding dimension (e.g., 128).
            head_dim (int): Output dimension of this single attention head (e.g., 32).
            context_size (int): Maximum sequence length (block_size, e.g., 128).
            init_std (float): Standard deviation for Gaussian weight initialization.

        Raises:
            ValueError: If d_model <= 0, head_dim <= 0, context_size <= 0, or init_std <= 0.
        """
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}.")
        if context_size <= 0:
            raise ValueError(f"context_size must be positive, got {context_size}.")
        if init_std <= 0:
            raise ValueError(f"init_std must be positive, got {init_std}.")

        self.d_model: int = d_model
        self.head_dim: int = head_dim
        self.context_size: int = context_size
        self.scale: float = 1.0 / math.sqrt(head_dim)

        # Learned linear projections without bias (standard GPT convention)
        self.query_proj: nn.Linear = nn.Linear(d_model, head_dim, bias=False)
        self.key_proj: nn.Linear = nn.Linear(d_model, head_dim, bias=False)
        self.value_proj: nn.Linear = nn.Linear(d_model, head_dim, bias=False)

        # Register lower-triangular causal mask as a non-trainable buffer
        # Moves automatically with .to(device) alongside model parameters
        causal_mask: torch.Tensor = torch.tril(
            torch.ones((context_size, context_size), dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask)

        self._init_weights(std=init_std)

    def _init_weights(self, std: float) -> None:
        """Initializes Q, K, and V projection matrices from N(0, std^2)."""
        nn.init.normal_(self.query_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.key_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.value_proj.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes causal self-attention for a batch of input sequences.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Contextualized head output of shape
                          (batch_size, seq_len, head_dim).

        Raises:
            TypeError: If x is not a floating-point tensor.
            ValueError: If x is not 3D, its feature dimension != d_model,
                        or seq_len > context_size.
        """
        if not torch.is_floating_point(x):
            raise TypeError(f"Expected floating-point tensor, got {x.dtype}.")
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input (batch_size, seq_len, d_model), got {tuple(x.shape)}."
            )

        _, seq_len, emb_dim = x.shape

        if emb_dim != self.d_model:
            raise ValueError(
                f"Input dimension ({emb_dim}) does not match d_model ({self.d_model})."
            )
        if seq_len > self.context_size:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds context_size ({self.context_size})."
            )

        # Step 1: Project input x into Query, Key, and Value matrices
        # Each has shape: (batch_size, seq_len, head_dim)
        queries: torch.Tensor = self.query_proj(x)
        keys: torch.Tensor = self.key_proj(x)
        values: torch.Tensor = self.value_proj(x)

        # Step 2: Compute raw similarity scores Q @ K^T and scale by 1 / sqrt(head_dim)
        # keys.transpose(-2, -1) has shape: (batch_size, head_dim, seq_len)
        # scores has shape: (batch_size, seq_len, seq_len)
        scores: torch.Tensor = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale

        # Step 3: Apply causal mask so position i cannot attend to any position j > i
        active_mask: torch.Tensor = self.causal_mask[:seq_len, :seq_len]
        masked_scores: torch.Tensor = scores.masked_fill(~active_mask, float("-inf"))

        # Step 4: Normalize scores into attention probabilities summing to 1.0
        # Shape: (batch_size, seq_len, seq_len)
        attention_weights: torch.Tensor = F.softmax(masked_scores, dim=-1)

        # Step 5: Compute weighted sum of Value vectors
        # Output shape: (batch_size, seq_len, head_dim)
        head_output: torch.Tensor = torch.matmul(attention_weights, values)
        return head_output

    @property
    def num_parameters(self) -> int:
        """Returns total learnable parameters in this single attention head."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, head_dim={self.head_dim}, "
            f"context_size={self.context_size}"
        )