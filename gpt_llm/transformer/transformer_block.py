import torch
import torch.nn as nn
from gpt_llm.transformer.feed_forward import FeedForward
from gpt_llm.transformer.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    Pre-Norm Decoder-Only Transformer Block.

    Combines Layer Normalization, Causal Multi-Head Self-Attention, Position-Wise
    Feed-Forward Network, and Residual Skip Connections into one complete,
    stackable Transformer layer:
        x = x + MultiHeadAttention(LayerNorm1(x))
        x = x + FeedForward(LayerNorm2(x))

    Preserves tensor shape (batch_size, seq_len, d_model) from input to output
    so multiple blocks can be stacked sequentially.

    Attributes:
        d_model (int): Model embedding dimensionality.
        num_heads (int): Number of parallel attention heads.
        context_size (int): Maximum sequence length supported by attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        context_size: int,
        expansion_factor: int = 4,
        init_std: float = 0.02,
    ) -> None:
        """
        Initializes all four core components of the Transformer block.

        Args:
            d_model (int): Model embedding dimension (e.g., 128).
            num_heads (int): Number of parallel attention heads (e.g., 4).
            context_size (int): Maximum sequence length (block_size, e.g., 128).
            expansion_factor (int): Hidden expansion multiplier for FeedForward.
            init_std (float): Standard deviation for Gaussian weight initialization.
        """
        super().__init__()

        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.context_size: int = context_size

        # Sub-layer 1 components: Pre-Norm + Masked Multi-Head Self-Attention
        self.ln1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.attention: MultiHeadAttention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            context_size=context_size,
            init_std=init_std,
        )

        # Sub-layer 2 components: Pre-Norm + Position-Wise Feed-Forward Network
        self.ln2: nn.LayerNorm = nn.LayerNorm(d_model)
        self.ffn: FeedForward = FeedForward(
            d_model=d_model,
            expansion_factor=expansion_factor,
            init_std=init_std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes position-aware embeddings through the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Contextualized output tensor of identical shape
                          (batch_size, seq_len, d_model).
        """
        # Sub-layer 1: Normalize -> Multi-Head Attention -> Add Residual Connection
        norm1_out: torch.Tensor = self.ln1(x)
        attn_out: torch.Tensor = self.attention(norm1_out)
        x = x + attn_out

        # Sub-layer 2: Normalize -> Feed-Forward Network -> Add Residual Connection
        norm2_out: torch.Tensor = self.ln2(x)
        ffn_out: torch.Tensor = self.ffn(norm2_out)
        x = x + ffn_out

        return x

    @property
    def num_parameters(self) -> int:
        """Returns total learnable parameters in this Transformer block."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, "
            f"context_size={self.context_size}"
        )


if __name__ == "__main__":
    # Quick end-to-end verification of the assembled TransformerBlock
    batch_size: int = 2
    seq_len: int = 16
    d_model: int = 128
    num_heads: int = 4
    context_size: int = 128

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        context_size=context_size,
    )
    print("Assembled Transformer Block:\n", block)
    print(f"Total Block Parameters: {block.num_parameters:,}")

    # Simulated output from EmbeddingTable + PositionalEmbedding: (2, 16, 128)
    sample_embeddings = torch.randn((batch_size, seq_len, d_model), dtype=torch.float32)
    output = block(sample_embeddings)

    print("Input Shape :", tuple(sample_embeddings.shape))
    print("Output Shape:", tuple(output.shape))