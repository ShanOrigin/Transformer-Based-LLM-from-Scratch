from typing import List
import torch
import torch.nn as nn
from gpt_llm.transformer.self_attention_head import SelfAttentionHead


class MultiHeadAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention Layer.

    Runs `num_heads` independent SelfAttentionHead modules in parallel across
    distinct subspaces of size `head_dim = d_model // num_heads`, concatenates
    their outputs back to `d_model`, and mixes them through an output linear
    projection matrix W_o.

    Attributes:
        d_model (int): Total model embedding dimensionality.
        num_heads (int): Number of parallel attention heads.
        head_dim (int): Dimensionality per individual attention head.
        heads (nn.ModuleList): Container holding `num_heads` SelfAttentionHead instances.
        out_proj (nn.Linear): Output mixing projection of shape (d_model, d_model).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        context_size: int,
        init_std: float = 0.02,
    ) -> None:
        """
        Initializes parallel attention heads and the final output projection.

        Args:
            d_model (int): Model embedding dimension (must be divisible by num_heads).
            num_heads (int): Number of parallel self-attention heads (e.g., 4).
            context_size (int): Maximum sequence length (block_size, e.g., 128).
            init_std (float): Standard deviation for Gaussian weight initialization.

        Raises:
            ValueError: If d_model or num_heads <= 0, or d_model % num_heads != 0.
        """
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be evenly divisible by num_heads ({num_heads})."
            )

        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.head_dim: int = d_model // num_heads

        # Build `num_heads` independent causal self-attention heads
        head_list: List[SelfAttentionHead] = []
        for _ in range(num_heads):
            head_list.append(
                SelfAttentionHead(
                    d_model=d_model,
                    head_dim=self.head_dim,
                    context_size=context_size,
                    init_std=init_std,
                )
            )
        self.heads: nn.ModuleList = nn.ModuleList(head_list)

        # Output linear projection W_o to blend information across all heads
        self.out_proj: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes multi-head self-attention on input embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Attention output tensor of shape (batch_size, seq_len, d_model).
        """
        # Step 1: Run each head independently -> list of (batch_size, seq_len, head_dim)
        head_outputs: List[torch.Tensor] = []
        for head in self.heads:
            head_outputs.append(head(x))

        # Step 2: Concatenate all head outputs along the last (feature) dimension
        # Shape becomes: (batch_size, seq_len, d_model)
        concatenated: torch.Tensor = torch.cat(head_outputs, dim=-1)

        # Step 3: Project concatenated features through W_o
        # Output shape: (batch_size, seq_len, d_model)
        return self.out_proj(concatenated)

    @property
    def num_parameters(self) -> int:
        """Returns total learnable parameters across all heads and output projection."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}"
        )