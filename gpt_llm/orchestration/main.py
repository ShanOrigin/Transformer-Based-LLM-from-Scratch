
from gpt_llm.config import IS_DEBUG
from gpt_llm.embeddings.positional_embedding import PositionalEmbedding
from gpt_llm.user_input.user_input import get_user_input
from gpt_llm.tokenization.tokenizer.byte_pair_level_regex_tokenizer import BytePairLevelRegexTokenizer
from gpt_llm.parameters import CONTEXT_SIZE, D_MODEL, VOCAB_SIZE
from gpt_llm.embeddings.embedding_table import EmbeddingTable
import torch 

def main() -> None:
    """Run input, text processing, chat templating, and tokenization."""
    try:
        # ------ Step 01: Capture raw user input. ------
        user_input = get_user_input()

        if IS_DEBUG: 
            print(f"\nUser Input : \n  {user_input} \n")

        # ------ Step 02: Tokenize the user input. ------

        #initialize the tokenizer
        tokenizer = BytePairLevelRegexTokenizer()

        if IS_DEBUG: 
            print("Tokenizer initialized.")
            print("VOCAB_SIZE : before loading files -> ", tokenizer.vocab_size)

        # load the tokenizer with the vocab and merges files 
        tokenizer.load("gpt_llm/tokenization/tokenizer/output_files/english_tokenizer_vocab.json",
                       "gpt_llm/tokenization/tokenizer/output_files/english_tokenizer_merges.json")

        if IS_DEBUG: 
            print("VOCAB_SIZE : after loading files -> ", tokenizer.vocab_size)

        # tokenize the user input
        tokens = tokenizer.encode(user_input)

        if IS_DEBUG: 
            print(f"\nTokenized Input : \n  {tokens} \n")
            print(f"Tokenized Input Length : {len(tokens)} \n")

        # decode the tokens (if de is enabled)
        if IS_DEBUG:
            decoded_input = tokenizer.decode(tokens)
            print(f"\nDecoded Input : \n  {decoded_input} \n")


        #------ Step 03: Create an embedding table and get embeddings for the tokenized input. ------
        embedding_table = EmbeddingTable(vocab_size=VOCAB_SIZE, d_model=D_MODEL)

        if IS_DEBUG:
            print("Embedding Table initialized.")
            print(f"Learnable Parameters: {embedding_table.num_parameters:,}")

        tokens_embeddings_tensors = embedding_table(torch.tensor(tokens).unsqueeze(0))

        if IS_DEBUG:
            print(f"Token Embeddings Shape: {tokens_embeddings_tensors.shape}")
            
        # ------ Step 3.5: Positional embedding table
        positional_embedding_table = PositionalEmbedding(context_size=CONTEXT_SIZE, d_model=D_MODEL)

        if IS_DEBUG:
            print("Positional Embedding Table initialized.")
            print(f"Learnable Parameters: {positional_embedding_table.num_parameters:,}")

        combined_embeddings = positional_embedding_table(tokens_embeddings_tensors)

        if IS_DEBUG:
            print(f"Combined Embeddings Shape: {combined_embeddings.shape}")

    except (TypeError, ValueError, FileNotFoundError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()

# to run this script use python -m gpt_llm.orchestration.main
