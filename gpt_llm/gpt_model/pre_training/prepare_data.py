from gpt_llm.config import CACHE_PATH , MERGES_PATH , VOCAB_PATH , CORPUS_PATH
from gpt_llm.gpt_model.pre_training.data_preparation import prepare_token_tensor
from gpt_llm.tokenization.tokenizer.byte_pair_level_regex_tokenizer import (
    BytePairLevelRegexTokenizer,
)



tokenizer = BytePairLevelRegexTokenizer()
tokenizer.load(vocab_file=VOCAB_PATH, merges_file=MERGES_PATH)

raw_text = CORPUS_PATH.read_text(encoding="utf-8")
encoded_ids = tokenizer.encode(raw_text)

token_tensor = prepare_token_tensor(token_ids=encoded_ids, save_path=CACHE_PATH)
print(f"Encoded & Cached {token_tensor.size(0):,} tokens -> {CACHE_PATH}")



# for data preparation and caching this this file before pre traning of model 
# python -m gpt_llm.gpt_model.pre_training.prepare_data