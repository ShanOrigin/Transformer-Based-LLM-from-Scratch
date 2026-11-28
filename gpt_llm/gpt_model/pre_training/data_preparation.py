from pathlib import Path
from typing import Sequence, Tuple
import torch


def prepare_token_tensor(
    token_ids: Sequence[int] | torch.Tensor,
    save_path: str | Path | None = None,
) -> torch.Tensor:
    """
    Converts a flat sequence of integer token IDs into a contiguous 1D
    `torch.long` tensor suitable for language model training, and optionally
    caches it to disk so the corpus does not need to be re-tokenized every run.

    Args:
        token_ids (Sequence[int] | torch.Tensor): Flat sequence of encoded corpus token IDs.
        save_path (str | Path | None): Optional `.pt` file path to cache the 1D tensor.

    Returns:
        torch.Tensor: 1D integer tensor of shape (num_tokens,) with dtype `torch.long`.

    Raises:
        ValueError: If `token_ids` is empty or not 1-dimensional.
    """
    if isinstance(token_ids, torch.Tensor):
        if token_ids.ndim != 1:
            raise ValueError(
                f"Expected a 1D tensor of token IDs, got shape {tuple(token_ids.shape)}."
            )
        tensor_ids: torch.Tensor = token_ids.to(dtype=torch.long).contiguous()
    else:
        if len(token_ids) == 0:
            raise ValueError("Cannot create token tensor from an empty sequence.")
        tensor_ids = torch.tensor(token_ids, dtype=torch.long)

    if save_path is not None:
        cache_file = Path(save_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor_ids, cache_file)

    return tensor_ids


def load_cached_tokens(file_path: str | Path) -> torch.Tensor:
    """
    Loads a previously saved 1D token ID tensor from disk.

    Args:
        file_path (str | Path): Path to the cached `.pt` token tensor file.

    Returns:
        torch.Tensor: 1D `torch.long` tensor of corpus token IDs.

    Raises:
        FileNotFoundError: If `file_path` does not exist.
        ValueError: If the loaded object is not a 1D tensor.
    """
    cache_path = Path(file_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cached token tensor not found: {cache_path}")

    loaded = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, torch.Tensor) or loaded.ndim != 1:
        raise ValueError("Loaded artifact must be a 1D PyTorch tensor.")

    return loaded.to(dtype=torch.long)


def split_train_val(
    token_stream: torch.Tensor,
    train_ratio: float = 0.9,
    context_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Splits a 1D corpus token stream chronologically into training and validation
    splits without shuffling, preserving natural document/sentence flow.

    Args:
        token_stream (torch.Tensor): 1D `torch.long` tensor of all corpus token IDs.
        train_ratio (float): Fraction of tokens allocated to training (0.0 < ratio < 1.0).
        context_size (int): Maximum sequence length used to verify both splits have
                            enough tokens for at least one `(x, y)` window.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - train_ids: 1D tensor containing the first `train_ratio` portion of tokens.
            - val_ids: 1D tensor containing the remaining `1 - train_ratio` tokens.

    Raises:
        ValueError: If `token_stream` is not 1D, `train_ratio` is outside (0, 1),
                    or either resulting split is too small for `context_size + 1`.
    """
    if token_stream.ndim != 1:
        raise ValueError(
            f"token_stream must be a 1D tensor, got shape {tuple(token_stream.shape)}."
        )
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(
            f"train_ratio must be strictly between 0.0 and 1.0, got {train_ratio}."
        )

    total_tokens: int = token_stream.size(0)
    split_index: int = int(total_tokens * train_ratio)

    train_ids: torch.Tensor = token_stream[:split_index]
    val_ids: torch.Tensor = token_stream[split_index:]

    min_required: int = context_size + 1
    if train_ids.size(0) <= min_required:
        raise ValueError(
            f"Training split ({train_ids.size(0)} tokens) must exceed "
            f"context_size + 1 ({min_required})."
        )
    if val_ids.size(0) <= min_required:
        raise ValueError(
            f"Validation split ({val_ids.size(0)} tokens) must exceed "
            f"context_size + 1 ({min_required})."
        )

    return train_ids, val_ids


def get_random_batch(
    token_stream: torch.Tensor,
    context_size: int,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Samples `batch_size` random contiguous windows of length `context_size`
    along with their 1-token right-shifted targets for Causal Language Modeling:
        x_batch[b] = token_stream[start : start + context_size]
        y_batch[b] = token_stream[start + 1 : start + context_size + 1]

    Args:
        token_stream (torch.Tensor): 1D `torch.long` tensor (either `train_ids` or `val_ids`).
        context_size (int): Number of tokens per training sequence (block_size).
        batch_size (int): Number of independent sequences stacked in parallel.
        device (torch.device | str): Target device ('cpu' or 'cuda') for output batches.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - x_batch: Input token IDs of shape `(batch_size, context_size)`.
            - y_batch: Target next-token IDs of shape `(batch_size, context_size)`.

    Raises:
        ValueError: If `batch_size <= 0`, `context_size <= 0`, or `token_stream` is too short.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if context_size <= 0:
        raise ValueError(f"context_size must be positive, got {context_size}.")

    max_start_offset: int = token_stream.size(0) - context_size - 1
    if max_start_offset <= 0:
        raise ValueError(
            f"token_stream length ({token_stream.size(0)}) must exceed "
            f"context_size + 1 ({context_size + 1})."
        )

    target_device = torch.device(device)

    # Draw `batch_size` random starting indices in [0, max_start_offset)
    start_indices: torch.Tensor = torch.randint(
        low=0,
        high=max_start_offset,
        size=(batch_size,),
    )

    x_chunks = []
    y_chunks = []
    for start in start_indices.tolist():
        x_chunks.append(token_stream[start : start + context_size])
        y_chunks.append(token_stream[start + 1 : start + context_size + 1])

    x_batch: torch.Tensor = torch.stack(x_chunks, dim=0).to(
        dtype=torch.long, device=target_device
    )
    y_batch: torch.Tensor = torch.stack(y_chunks, dim=0).to(
        dtype=torch.long, device=target_device
    )

    return x_batch, y_batch
