# =====================================================================
# 1. TOKENIZATION & MODEL ARCHITECTURE HYPERPARAMETERS
# =====================================================================
VOCAB_SIZE = 8192           # 2^13: Number of unique tokens in vocabulary
D_MODEL = 128               # 2^7: Embedding dimension per token
CONTEXT_SIZE = 128          # 2^7: Maximum sequence length (block_size)
NUM_HEADS = 4               # 2^2: Parallel attention heads (head_dim = 128 // 4 = 32)
NUM_LAYERS = 4              # 2^2: Number of stacked Transformer decoder blocks
EXPANSION_FACTOR = 4        # 2^2: Feed-Forward hidden width multiplier (128 * 4 = 512)
INIT_STD = 0.02             # Standard deviation for Gaussian weight initialization


# =====================================================================
# 2. PRE-TRAINING DATA & OPTIMIZATION HYPERPARAMETERS
# =====================================================================
TRAIN_SPLIT = 0.9           # 90% training corpus, 10% validation corpus
BATCH_SIZE = 16             # 2^4: Sequences processed per step (CPU-friendly)

EPOCHS = 5                  # Total passes through the training schedule
STEPS_PER_EPOCH = 500       # Number of gradient update steps per epoch (5 * 500 = 2,500 steps)
LEARNING_RATE = 3e-4        # Standard AdamW peak learning rate for small GPT models
WEIGHT_DECAY = 0.01         # L2 regularization coefficient for AdamW
MAX_GRAD_NORM = 1.0         # Gradient clipping threshold to prevent exploding gradients


# =====================================================================
# 3. PRE-TRAINING EVALUATION, LOGGING & CHECKPOINTING
# =====================================================================
EVAL_INTERVAL = 100         # Evaluate validation loss every N training steps
EVAL_BATCHES = 20           # Number of random batches averaged during loss estimation
PROGRESS_INTERVAL = 25      # Print training progress bar every N steps
RANDOM_SEED = 42            # Fixed seed for reproducible data splits and batch sampling