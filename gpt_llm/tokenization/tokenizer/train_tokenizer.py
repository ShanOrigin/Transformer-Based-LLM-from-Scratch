from pathlib import Path
from gpt_llm.parameters import VOCAB_SIZE
from gpt_llm.tokenization.tokenizer.byte_pair_level_regex_tokenizer import BytePairLevelRegexTokenizer

# Points to D:\AI\Learning\gpt_llm\tokenization
TOKENIZATION_DIR = Path(__file__).resolve().parent.parent

corpus = TOKENIZATION_DIR / "corpus" / "merged_corpus.txt"
save_dir = TOKENIZATION_DIR / "tokenizer" / "output_files" / "english_tokenizer"

if __name__ == "__main__":
    # Ensure output_files directory exists before saving
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = BytePairLevelRegexTokenizer()
    corpus_ids = tokenizer.fit_transform(corpus_file=corpus, vocab_size=VOCAB_SIZE)
    vocab_file, merges_file = tokenizer.save(str(save_dir))

    # Load into a fresh instance and test round-trip encoding/decoding
    restored = BytePairLevelRegexTokenizer()
    restored.load(vocab_file, merges_file)

    prompt = (
        "BytePairLevelRegexTokenizer works cleanly in 2026!... I hope it does! "
        "it's a good day to test it out. Let's see if it can handle this prompt correctly."
    )
    encoded = restored.encode(prompt)
    decoded = restored.decode(encoded)

    print("Encoded:", encoded)
    print("Decoded:", decoded)
    assert prompt == decoded


## python -m gpt_llm.tokenization.tokenizer.train_tokenizer ,
#  use this command to run the script from the root directory of the project.