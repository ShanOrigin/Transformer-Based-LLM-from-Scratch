import torch
import torch.nn as nn


class EmbeddingTable(nn.Module):
    """
    Learnable Token Embedding Table for converting discrete token IDs into
    dense continuous vector representations.

    Maps a 2D batch of integer token IDs with shape (batch_size, seq_len)
    into a 3D floating-point tensor with shape (batch_size, seq_len, d_model)
    via direct row lookup.

    Attributes:
        vocab_size (int): Total number of unique tokens in the vocabulary.
        d_model (int): Dimensionality of each token's dense embedding vector.
        table (nn.Embedding): Underlying learnable lookup matrix of shape
                              (vocab_size, d_model).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        init_std: float = 0.02,
    ) -> None:
        """
        Initializes the token embedding lookup table with Gaussian weights.

        Args:
            vocab_size (int): Size of the tokenizer vocabulary (e.g., 8192).
            d_model (int): Embedding vector dimension per token (e.g., 128).
            init_std (float): Standard deviation for normal weight initialization.
                              Defaults to 0.02 (standard GPT initialization).

        Raises:
            ValueError: If vocab_size <= 0, d_model <= 0, or init_std <= 0.
        """
        super().__init__()

        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if init_std <= 0:
            raise ValueError(f"init_std must be positive, got {init_std}.")

        self.vocab_size: int = vocab_size
        self.d_model: int = d_model

        # Create the learnable lookup matrix: shape (vocab_size, d_model)
        self.table: nn.Embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model,
        )

        self._init_weights(std=init_std)

    def _init_weights(self, std: float) -> None:
        """
        Initializes embedding table weights from a zero-mean normal distribution
        N(0, std^2) for stable gradient flow during early training.

        Args:
            std (float): Standard deviation of the normal distribution.
        """
        nn.init.normal_(self.table.weight, mean=0.0, std=std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Looks up dense embedding vectors for a batch of integer token ID sequences.

        Args:
            token_ids (torch.Tensor): 2D integer tensor of token IDs with shape
                                      (batch_size, seq_len). Values must lie in
                                      [0, vocab_size - 1].

        Returns:
            torch.Tensor: 3D float tensor of token embeddings with shape
                          (batch_size, seq_len, d_model).

        Raises:
            TypeError: If token_ids is not a PyTorch integer tensor.
            ValueError: If token_ids is not 2D or contains IDs outside [0, vocab_size - 1].
        """
        if token_ids.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint8):
            raise TypeError(
                f"token_ids must be an integer tensor (e.g., torch.long), got {token_ids.dtype}."
            )

        if token_ids.ndim != 2:
            raise ValueError(
                f"Expected 2D input tensor (batch_size, seq_len), got shape {tuple(token_ids.shape)}."
            )

        if token_ids.numel() > 0:
            min_id = int(token_ids.min().item())
            max_id = int(token_ids.max().item())
            if min_id < 0 or max_id >= self.vocab_size:
                raise ValueError(
                    f"Token IDs must be within [0, {self.vocab_size - 1}], "
                    f"got range [{min_id}, {max_id}]."
                )

        # nn.Embedding expects torch.long (int64) indices
        if token_ids.dtype != torch.long:
            token_ids = token_ids.to(torch.long)

        return self.table(token_ids)

    @property
    def num_parameters(self) -> int:
        """Returns the total number of learnable parameters in the embedding table."""
        return self.vocab_size * self.d_model

    def extra_repr(self) -> str:
        """Provides a clean summary string when printing the module."""
        return f"vocab_size={self.vocab_size}, d_model={self.d_model}"
