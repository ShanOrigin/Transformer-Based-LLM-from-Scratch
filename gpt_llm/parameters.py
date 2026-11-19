
# --- Tokenization & Architecture Dimensions ---
VOCAB_SIZE = 8192       # 2^13: Number of unique tokens in vocabulary
D_MODEL = 128           # 2^7: Embedding dimension per token
CONTEXT_SIZE = 128      # 2^7: Maximum sequence length (block_size)

# --- Training & Data Pipeline ---
BATCH_SIZE = 16         # 2^4: Number of sequences processed in parallel (CPU-friendly)
TRAIN_SPLIT = 0.9       # 90% training, 10% validation