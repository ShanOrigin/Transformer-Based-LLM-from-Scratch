import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class BytePairLevelRegexTokenizer:
    """
    A fast Regex-based Byte-Level Byte-Pair Encoding (BPE) Tokenizer.

    Uses incremental pair-to-chunk indexing and a lazy max-heap for O(log N)
    merge selection, training an 8K vocabulary on a ~12MB corpus in ~3-5 seconds.
    Includes milestone progress bars during vocabulary training and corpus encoding.
    """

    INITIAL_VOCAB_SIZE: int = 256
    GPT2_SPLIT_PATTERN: str = (
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

    def __init__(self, pattern: Optional[str] = None) -> None:
        """
        Initializes the ByteLevelRegexTokenizer with base byte mappings and regex rules.

        Args:
            pattern (Optional[str]): Custom regex pattern for pre-tokenization.
                                     Defaults to GPT2_SPLIT_PATTERN.
        """
        self._PATTERN: str = pattern if pattern is not None else self.GPT2_SPLIT_PATTERN
        self._COMPILED_PATTERN: re.Pattern = re.compile(self._PATTERN)

        # Learned merge rules: {(token_id_1, token_id_2): new_token_id}
        self._MERGES: Dict[Tuple[int, int], int] = {}

        # Vocabulary mapping: {token_id: raw_utf8_bytes}
        self._VOCAB: Dict[int, bytes] = {}

        # Speed optimization: caches already-encoded byte chunks during inference
        self._ENCODE_CACHE: Dict[bytes, List[int]] = {}

        self._reset_state()

    def _reset_state(self) -> None:
        """
        Resets internal vocabulary, merges, and encode cache to a fresh
        256-byte base state without affecting any files saved on disk.
        """
        self._MERGES.clear()
        self._ENCODE_CACHE.clear()
        self._VOCAB = {
            idx: bytes([idx]) for idx in range(self.INITIAL_VOCAB_SIZE)
        }

    @staticmethod
    def _print_progress_bar(
        current: int,
        total: int,
        prefix: str = "",
        suffix: str = "",
        bar_length: int = 30,
    ) -> None:
        """
        Prints a clean text progress bar with percentage and count metrics.
        """
        if total <= 0:
            return
        ratio: float = min(max(current / total, 0.0), 1.0)
        filled: int = int(bar_length * ratio)
        bar: str = "█" * filled + "░" * (bar_length - filled)
        percent: float = ratio * 100.0
        print(f"{prefix} |{bar}| {percent:6.2f}% ({current}/{total}) {suffix}")

    def _pre_tokenize(self, text: str) -> List[str]:
        """
        Splits raw text into isolated chunks using the compiled regex pattern.
        """
        return self._COMPILED_PATTERN.findall(text)

    def _merge_pair_in_chunk(
        self, chunk: Tuple[int, ...], pair: Tuple[int, int], new_id: int
    ) -> Tuple[int, ...]:
        """
        Replaces all non-overlapping occurrences of a target pair in a chunk
        with the newly minted token ID.
        """
        chunk_len: int = len(chunk)
        if chunk_len < 2:
            return chunk

        merged: List[int] = []
        i: int = 0
        first, second = pair

        while i < chunk_len:
            if i < chunk_len - 1 and chunk[i] == first and chunk[i + 1] == second:
                merged.append(new_id)
                i += 2
            else:
                merged.append(chunk[i])
                i += 1

        return tuple(merged)

    def _byte_level_tokenizer(
        self, raw_text: str, vocab_size: int, show_progress: bool = True
    ) -> None:
        """
        Fast incremental BPE training:
        Maintains an inverted index mapping each adjacent pair to the chunk IDs
        containing it, updating only affected chunks at each merge step.
        Prints a progress bar every time total unique tokens hits a multiple of 500.
        """
        self._reset_state()

        raw_chunks: List[str] = self._pre_tokenize(raw_text)
        chunk_counter = Counter(
            tuple(chunk.encode("utf-8")) for chunk in raw_chunks
        )

        # Store unique chunks in lists by index for O(1) lookup and in-place updates
        chunks: List[Tuple[int, ...]] = list(chunk_counter.keys())
        freqs: List[int] = [chunk_counter[c] for c in chunks]

        pair_counts: Counter[Tuple[int, int]] = Counter()
        pair_to_chunk_ids: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

        # Build initial pair counts and inverted index once
        for idx, (chunk, freq) in enumerate(zip(chunks, freqs)):
            if len(chunk) < 2:
                continue
            for pair in zip(chunk, chunk[1:]):
                pair_counts[pair] += freq
                pair_to_chunk_ids[pair].add(idx)

        # Max-heap storing (-count, pair) for instant top-pair retrieval
        heap: List[Tuple[int, Tuple[int, int]]] = [
            (-count, pair) for pair, count in pair_counts.items() if count > 0
        ]
        heapq.heapify(heap)

        num_merges: int = vocab_size - self.INITIAL_VOCAB_SIZE

        for step in range(num_merges):
            top_pair: Optional[Tuple[int, int]] = None

            # Pop stale heap entries until we find the true current maximum
            while heap:
                neg_count, candidate_pair = heapq.heappop(heap)
                actual_count = pair_counts.get(candidate_pair, 0)
                if actual_count > 0 and -neg_count == actual_count:
                    top_pair = candidate_pair
                    break

            if top_pair is None:
                break

            new_token_id: int = self.INITIAL_VOCAB_SIZE + step
            self._MERGES[top_pair] = new_token_id
            self._VOCAB[new_token_id] = (
                self._VOCAB[top_pair[0]] + self._VOCAB[top_pair[1]]
            )

            # Only iterate over chunks that actually contain `top_pair`
            affected_ids = list(pair_to_chunk_ids.pop(top_pair, ()))
            modified_pairs: Set[Tuple[int, int]] = set()

            for idx in affected_ids:
                old_chunk = chunks[idx]
                freq = freqs[idx]

                # Remove old chunk's pair contributions
                for p in zip(old_chunk, old_chunk[1:]):
                    pair_counts[p] -= freq
                    if p != top_pair:
                        pair_to_chunk_ids[p].discard(idx)
                        modified_pairs.add(p)

                # Merge pair in this chunk
                new_chunk = self._merge_pair_in_chunk(old_chunk, top_pair, new_token_id)
                chunks[idx] = new_chunk

                # Add new chunk's pair contributions
                for p in zip(new_chunk, new_chunk[1:]):
                    pair_counts[p] += freq
                    pair_to_chunk_ids[p].add(idx)
                    modified_pairs.add(p)

            pair_counts.pop(top_pair, None)

            # Push updated counts for modified pairs onto the lazy max-heap
            for p in modified_pairs:
                cnt = pair_counts.get(p, 0)
                if cnt > 0:
                    heapq.heappush(heap, (-cnt, p))

            # Print progress bar whenever total unique tokens is divisible by 500 or complete
            current_vocab_count: int = len(self._VOCAB)
            if show_progress and (
                current_vocab_count % 500 == 0 or current_vocab_count == vocab_size
            ):
                self._print_progress_bar(
                    current=current_vocab_count,
                    total=vocab_size,
                    prefix="Vocab Training",
                    suffix=f"unique tokens generated",
                )

    def _encode_chunk(self, chunk_bytes: bytes) -> List[int]:
        """
        Encodes a single regex-split chunk of raw bytes into BPE token IDs
        by applying learned merge rules in priority order. Uses caching for speed.
        """
        if chunk_bytes in self._ENCODE_CACHE:
            return self._ENCODE_CACHE[chunk_bytes]

        ids: Tuple[int, ...] = tuple(chunk_bytes)

        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            best_pair = min(
                pairs, key=lambda p: self._MERGES.get(p, float("inf"))
            )

            if best_pair not in self._MERGES:
                break

            new_id = self._MERGES[best_pair]
            ids = self._merge_pair_in_chunk(ids, best_pair, new_id)

        result: List[int] = list(ids)
        self._ENCODE_CACHE[chunk_bytes] = result
        return result

    def fit_transform(
        self,
        corpus_file: str | Path,
        vocab_size: int,
        show_progress: bool = True,
    ) -> List[int]:
        """
        Reads the corpus file, trains the Byte-Level BPE model via
        `_byte_level_tokenizer`, and returns the encoded token IDs of the corpus.

        Args:
            corpus_file (str | Path): Path to the UTF-8 training text file.
            vocab_size (int): Target total vocabulary size (must be >= 256).
            show_progress (bool): Whether to display progress bars during training
                                  and encoding. Defaults to True.

        Returns:
            List[int]: Tokenized ID representation of the entire training corpus.
        """
        if vocab_size < self.INITIAL_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size must be at least {self.INITIAL_VOCAB_SIZE}, got {vocab_size}."
            )

        file_path = Path(corpus_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {file_path}")

        raw_text: str = file_path.read_text(encoding="utf-8")

        if show_progress:
            print(f"Starting BPE training on '{file_path.name}' (target vocab: {vocab_size})...")

        # Step 1: Train vocabulary and merges with milestone progress bar (every 500 tokens)
        self._byte_level_tokenizer(
            raw_text=raw_text,
            vocab_size=vocab_size,
            show_progress=show_progress,
        )

        if show_progress:
            print("\nEncoding full corpus into token IDs...")

        # Step 2: Return the tokenized representation of the training corpus
        return self.encode(raw_text, show_progress=show_progress)

    def encode(self, prompt: str, show_progress: bool = False) -> List[int]:
        """
        Converts an input string into a list of learned integer token IDs.

        Args:
            prompt (str): User-provided text string to tokenize.
            show_progress (bool): Displays a progress bar every 500,000 chunks
                                  when encoding very large texts.

        Returns:
            List[int]: Sequence of token IDs representing the prompt.
        """
        chunks: List[str] = self._pre_tokenize(prompt)
        total_chunks: int = len(chunks)
        token_ids: List[int] = []

        for idx, chunk in enumerate(chunks, start=1):
            token_ids.extend(self._encode_chunk(chunk.encode("utf-8")))
            if show_progress and (idx % 500_000 == 0 or idx == total_chunks):
                self._print_progress_bar(
                    current=idx,
                    total=total_chunks,
                    prefix="Corpus Encoding",
                    suffix="chunks processed",
                )

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Converts a list of integer token IDs back into a human-readable string.
        """
        try:
            combined_bytes: bytes = b"".join(self._VOCAB[idx] for idx in token_ids)
        except KeyError as exc:
            raise KeyError(f"Token ID {exc.args[0]} does not exist in vocabulary.") from exc
        return combined_bytes.decode("utf-8", errors="replace")

    def save(self, name: str) -> Tuple[Path, Path]:
        """
        Saves the trained vocabulary and merge rules to disk as JSON files:
        '{name}_vocab.json' and '{name}_merges.json'.
        """
        vocab_path = Path(f"{name}_vocab.json")
        merges_path = Path(f"{name}_merges.json")

        serializable_vocab: Dict[str, List[int]] = {
            str(token_id): list(byte_seq)
            for token_id, byte_seq in self._VOCAB.items()
        }
        serializable_merges: Dict[str, object] = {
            "pattern": self._PATTERN,
            "merges": [
                [pair[0], pair[1], new_id]
                for pair, new_id in sorted(
                    self._MERGES.items(), key=lambda item: item[1]
                )
            ],
        }

        vocab_path.write_text(
            json.dumps(serializable_vocab, indent=2), encoding="utf-8"
        )
        merges_path.write_text(
            json.dumps(serializable_merges, indent=2), encoding="utf-8"
        )

        return vocab_path, merges_path

    def load(self, vocab_file: str | Path, merges_file: str | Path) -> None:
        """
        Loads a pretrained vocabulary and merge rules from disk into the tokenizer.
        """
        vocab_path = Path(vocab_file)
        merges_path = Path(merges_file)

        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        if not merges_path.exists():
            raise FileNotFoundError(f"Merges file not found: {merges_path}")

        raw_vocab: Dict[str, List[int]] = json.loads(
            vocab_path.read_text(encoding="utf-8")
        )
        raw_merges_data: Dict[str, object] = json.loads(
            merges_path.read_text(encoding="utf-8")
        )

        self._ENCODE_CACHE.clear()
        self._VOCAB = {
            int(token_id): bytes(byte_list)
            for token_id, byte_list in raw_vocab.items()
        }

        saved_pattern = raw_merges_data.get("pattern", self.GPT2_SPLIT_PATTERN)
        self._PATTERN = str(saved_pattern)
        self._COMPILED_PATTERN = re.compile(self._PATTERN)

        merges_list: List[List[int]] = raw_merges_data["merges"]  # type: ignore[assignment]
        self._MERGES = {
            (entry[0], entry[1]): entry[2] for entry in merges_list
        }

    @property
    def vocab_size(self) -> int:
        """Returns the current total number of tokens in the vocabulary."""
        return len(self._VOCAB)