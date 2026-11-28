import math
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from gpt_llm.gpt_model.pre_training.data_preparation import (
    get_random_batch,
    load_cached_tokens,
    prepare_token_tensor,
    split_train_val,
)
from gpt_llm.parameters import (
    BATCH_SIZE,
    CONTEXT_SIZE,
    EPOCHS,
    EVAL_BATCHES,
    EVAL_INTERVAL,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    PROGRESS_INTERVAL,
    RANDOM_SEED,
    STEPS_PER_EPOCH,
    TRAIN_SPLIT,
    WEIGHT_DECAY,
)
from gpt_llm.tokenization.tokenizer.byte_pair_level_regex_tokenizer import (
    BytePairLevelRegexTokenizer,
)


def print_progress_bar(
    current: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    bar_length: int = 28,
) -> None:
    """
    Prints a clean console progress bar with percentage and step counters.

    Args:
        current (int): Current completed step count.
        total (int): Total scheduled steps.
        prefix (str): Text displayed before the bar.
        suffix (str): Metrics displayed after the bar.
        bar_length (int): Character width of the visual bar.
    """
    if total <= 0:
        return

    ratio: float = min(max(current / total, 0.0), 1.0)
    filled: int = int(bar_length * ratio)
    bar: str = "█" * filled + "░" * (bar_length - filled)
    percent: float = ratio * 100.0
    print(f"{prefix} |{bar}| {percent:6.2f}% ({current}/{total}) {suffix}")


def get_cosine_warmup_lr(
    current_step: int,
    total_steps: int,
    max_lr: float,
    min_lr_ratio: float = 0.1,
    warmup_ratio: float = 0.05,
) -> float:
    """
    Computes the learning rate at `current_step` using:
      1. Linear warmup from 0 -> `max_lr` during the first `warmup_ratio` of steps.
      2. Cosine decay from `max_lr` -> `max_lr * min_lr_ratio` for remaining steps.

    Args:
        current_step (int): 1-based current optimization step.
        total_steps (int): Total optimization steps across all epochs.
        max_lr (float): Peak learning rate (e.g., 3e-4).
        min_lr_ratio (float): Final minimum LR fraction of `max_lr`.
        warmup_ratio (float): Fraction of total steps spent warming up.

    Returns:
        float: Scheduled learning rate for `current_step`.
    """
    min_lr: float = max_lr * min_lr_ratio
    warmup_steps: int = max(1, int(total_steps * warmup_ratio))

    # Phase 1: Linear Warmup (prevents early gradient shocks on random weights)
    if current_step <= warmup_steps:
        return max_lr * (current_step / warmup_steps)

    # Phase 2: Cosine Annealing Decay
    progress: float = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine_decay: float = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_decay


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    context_size: int = CONTEXT_SIZE,
    batch_size: int = BATCH_SIZE,
    eval_batches: int = EVAL_BATCHES,
) -> Dict[str, float]:
    """
    Evaluates average Cross-Entropy loss on both training and validation splits
    without tracking gradients (`torch.no_grad()` + `model.eval()`).

    Args:
        model (nn.Module): GPTModel accepting `(x_batch, targets=y_batch)`.
        train_ids (torch.Tensor): 1D training token tensor.
        val_ids (torch.Tensor): 1D validation token tensor.
        context_size (int): Sequence window length.
        batch_size (int): Evaluation batch size.
        eval_batches (int): Number of random batches averaged per split.

    Returns:
        Dict[str, float]: Dictionary with keys `'train_loss'`, `'val_loss'`,
                          and `'val_perplexity'`.
    """
    was_training: bool = model.training
    model.eval()

    device: torch.device = next(model.parameters()).device
    splits: Dict[str, torch.Tensor] = {"train_loss": train_ids, "val_loss": val_ids}
    metrics: Dict[str, float] = {}

    for metric_name, stream in splits.items():
        total_loss: float = 0.0
        for _ in range(eval_batches):
            x_batch, y_batch = get_random_batch(
                token_stream=stream,
                context_size=context_size,
                batch_size=batch_size,
                device=device,
            )
            _, loss = model(x_batch, targets=y_batch)
            total_loss += float(loss.item())

        metrics[metric_name] = total_loss / eval_batches

    # Perplexity = exp(val_loss); intuitive measure of effective vocabulary uncertainty
    metrics["val_perplexity"] = math.exp(min(metrics["val_loss"], 20.0))

    if was_training:
        model.train()

    return metrics


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    max_grad_norm: float = MAX_GRAD_NORM,
) -> float:
    """
    Executes one single gradient descent update:
      1. Clears old gradients (`set_to_none=True` for lower CPU memory overhead).
      2. Forward pass + Cross-Entropy loss computation.
      3. Backpropagation (`loss.backward()`).
      4. Global gradient norm clipping (`max_grad_norm`).
      5. Parameter update (`optimizer.step()`).

    Args:
        model (nn.Module): GPTModel instance.
        optimizer (torch.optim.Optimizer): AdamW optimizer.
        x_batch (torch.Tensor): Input token IDs of shape `(batch_size, context_size)`.
        y_batch (torch.Tensor): Shifted target IDs of shape `(batch_size, context_size)`.
        max_grad_norm (float): Maximum allowed L2 gradient norm.

    Returns:
        float: Scalar training loss for this batch.
    """
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x_batch, targets=y_batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    return float(loss.item())


def _load_or_encode_corpus(
    corpus_file: str | Path,
    vocab_file: str | Path,
    merges_file: str | Path,
    cache_path: Optional[str | Path] = None,
) -> torch.Tensor:
    """
    Loads cached encoded corpus tokens if available on disk; otherwise loads the
    saved BPE tokenizer, encodes `corpus_file` into a 1D tensor, and caches it.
    """
    if cache_path is not None and Path(cache_path).exists():
        print(f"Loading cached tokenized corpus from '{cache_path}'...")
        return load_cached_tokens(cache_path)

    corpus_path = Path(corpus_file)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    print(f"Encoding corpus '{corpus_path.name}' with saved BPE tokenizer...")
    tokenizer = BytePairLevelRegexTokenizer()
    tokenizer.load(vocab_file=vocab_file, merges_file=merges_file)

    raw_text: str = corpus_path.read_text(encoding="utf-8")
    encoded_ids: List[int] = tokenizer.encode(raw_text)

    return prepare_token_tensor(token_ids=encoded_ids, save_path=cache_path)


def train_model(
    model: nn.Module,
    corpus_file: str | Path,
    vocab_file: str | Path,
    merges_file: str | Path,
    token_cache_file: Optional[str | Path] = None,
    checkpoint_save_path: Optional[str | Path] = None,
    epochs: int = EPOCHS,
    steps_per_epoch: int = STEPS_PER_EPOCH,
    batch_size: int = BATCH_SIZE,
    context_size: int = CONTEXT_SIZE,
    train_split: float = TRAIN_SPLIT,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    max_grad_norm: float = MAX_GRAD_NORM,
    eval_interval: int = EVAL_INTERVAL,
    eval_batches: int = EVAL_BATCHES,
    progress_interval: int = PROGRESS_INTERVAL,
    seed: int = RANDOM_SEED,
    device: Optional[torch.device | str] = None,
) -> Dict[str, object]:
    """
    End-to-end Pre-Training Pipeline for GPTModel.

    Takes a model and raw corpus path, handles tokenization/caching, performs the
    chronological train/validation split, configures AdamW + warmup/cosine LR
    scheduling, executes training with gradient clipping, runs periodic validation
    evaluations, and saves the best checkpoint automatically.

    Args:
        model (nn.Module): Initialized GPTModel instance.
        corpus_file (str | Path): Path to raw UTF-8 text corpus (e.g., `merged_corpus.txt`).
        vocab_file (str | Path): Path to saved `*_vocab.json`.
        merges_file (str | Path): Path to saved `*_merges.json`.
        token_cache_file (Optional[str | Path]): Optional `.pt` cache path for encoded corpus IDs.
        checkpoint_save_path (Optional[str | Path]): Optional path to save best model checkpoint.
        epochs (int): Number of training epochs (defaults to `PARAMETERS.EPOCHS`).
        steps_per_epoch (int): Gradient updates per epoch (`PARAMETERS.STEPS_PER_EPOCH`).
        batch_size (int): Sequences per step (`PARAMETERS.BATCH_SIZE`).
        context_size (int): Sequence length (`PARAMETERS.CONTEXT_SIZE`).
        train_split (float): Train/val split ratio (`PARAMETERS.TRAIN_SPLIT`).
        learning_rate (float): Peak AdamW learning rate (`PARAMETERS.LEARNING_RATE`).
        weight_decay (float): AdamW weight decay (`PARAMETERS.WEIGHT_DECAY`).
        max_grad_norm (float): Gradient clipping norm (`PARAMETERS.MAX_GRAD_NORM`).
        eval_interval (int): Run validation evaluation every N steps (`PARAMETERS.EVAL_INTERVAL`).
        eval_batches (int): Batches averaged per evaluation (`PARAMETERS.EVAL_BATCHES`).
        progress_interval (int): Print progress bar every N steps (`PARAMETERS.PROGRESS_INTERVAL`).
        seed (int): Random seed (`PARAMETERS.RANDOM_SEED`).
        device (Optional[torch.device | str]): Compute device ('cpu' or 'cuda').

    Returns:
        Dict[str, object]: Dictionary containing `'history'` (`train_loss`, `val_loss`,
                           `eval_steps`), `'best_val_loss'`, and `'optimizer'`.
    """
    torch.manual_seed(seed)

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    model.to(target_device)

    # Step 1: Load or encode corpus into a 1D tensor and split into train / val
    corpus_tensor: torch.Tensor = _load_or_encode_corpus(
        corpus_file=corpus_file,
        vocab_file=vocab_file,
        merges_file=merges_file,
        cache_path=token_cache_file,
    )
    train_ids, val_ids = split_train_val(
        token_stream=corpus_tensor,
        train_ratio=train_split,
        context_size=context_size,
    )
    train_ids = train_ids.to(target_device)
    val_ids = val_ids.to(target_device)

    # Step 2: Initialize AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_steps: int = epochs * steps_per_epoch
    current_step: int = 0
    best_val_loss: float = float("inf")

    history: Dict[str, List[float]] = {
        "step_train_loss": [],
        "eval_train_loss": [],
        "eval_val_loss": [],
        "eval_steps": [],
    }

    param_count: int = sum(p.numel() for p in model.parameters())
    print("\n" + "=" * 72)
    print(
        f"GPT Pre-Training | Device: {target_device} | Params: {param_count:,}\n"
        f"Train Tokens: {train_ids.size(0):,} | Val Tokens: {val_ids.size(0):,} | "
        f"Total Steps: {total_steps}"
    )
    print("=" * 72)

    # Step 3: Main Sequential Training Loop with Periodic Validation
    for epoch in range(1, epochs + 1):
        model.train()

        for _ in range(1, steps_per_epoch + 1):
            current_step += 1

            # Update learning rate according to Linear Warmup + Cosine Decay schedule
            scheduled_lr: float = get_cosine_warmup_lr(
                current_step=current_step,
                total_steps=total_steps,
                max_lr=learning_rate,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = scheduled_lr

            # Sample random 1-token shifted training batch
            x_batch, y_batch = get_random_batch(
                token_stream=train_ids,
                context_size=context_size,
                batch_size=batch_size,
                device=target_device,
            )

            # Execute forward pass, loss, backpropagation, gradient clipping & step
            step_loss: float = train_step(
                model=model,
                optimizer=optimizer,
                x_batch=x_batch,
                y_batch=y_batch,
                max_grad_norm=max_grad_norm,
            )
            history["step_train_loss"].append(step_loss)

            # Display training progress bar at configured intervals
            if current_step % progress_interval == 0 or current_step == total_steps:
                print_progress_bar(
                    current=current_step,
                    total=total_steps,
                    prefix=f"Epoch {epoch}/{epochs}",
                    suffix=f"lr: {scheduled_lr:.6f} | step_loss: {step_loss:.4f}",
                )

            # Pause briefly to evaluate Train vs. Val loss and save best checkpoint
            if current_step % eval_interval == 0 or current_step == total_steps:
                eval_metrics: Dict[str, float] = estimate_loss(
                    model=model,
                    train_ids=train_ids,
                    val_ids=val_ids,
                    context_size=context_size,
                    batch_size=batch_size,
                    eval_batches=eval_batches,
                )
                t_loss: float = eval_metrics["train_loss"]
                v_loss: float = eval_metrics["val_loss"]
                ppl: float = eval_metrics["val_perplexity"]

                history["eval_steps"].append(float(current_step))
                history["eval_train_loss"].append(t_loss)
                history["eval_val_loss"].append(v_loss)

                saved_note: str = ""
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    if checkpoint_save_path is not None and hasattr(model, "save"):
                        model.save(checkpoint_save_path)
                        saved_note = " | [Best Checkpoint Saved]"

                print(
                    f"  --> [Eval @ Step {current_step}/{total_steps}] "
                    f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | "
                    f"Perplexity: {ppl:.2f}{saved_note}"
                )

    print("=" * 72)
    print(f"Pre-Training Complete! Best Validation Loss: {best_val_loss:.4f}")
    print("=" * 72)

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "optimizer": optimizer,
    }