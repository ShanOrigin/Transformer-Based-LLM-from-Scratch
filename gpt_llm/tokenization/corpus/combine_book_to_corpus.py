from pathlib import Path

# Always points to D:\AI\LEARNING\gpt_llm\TOKENIZATION\corpus
BASE_DIR = Path(__file__).resolve().parent
input_dir = BASE_DIR / "books"
output_corpus = BASE_DIR / "merged_corpus.txt"

txt_files = sorted(input_dir.glob("*.txt"))
if not txt_files:
    raise FileNotFoundError(f"No .txt files found in: {input_dir}")

with output_corpus.open("w", encoding="utf-8") as outfile:
    for txt_file in txt_files:
        content = txt_file.read_text(encoding="utf-8", errors="ignore")

        # Strip Project Gutenberg header and footer boilerplate if present
        start_marker = "*** START OF"
        end_marker = "*** END OF"

        if start_marker in content:
            content = content.split(start_marker, 1)[1].split("\n", 1)[-1]
        if end_marker in content:
            content = content.split(end_marker, 1)[0]

        outfile.write(content.strip() + "\n\n")
        print(f"Merged: {txt_file.name}")

print(f"\nDone! Total corpus size: {output_corpus.stat().st_size / (1024 * 1024):.2f} MB")