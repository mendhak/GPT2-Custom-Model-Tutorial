from tokenise import BPE_token
from pathlib import Path
import os# the folder 'text' contains all the files
paths = [str(x) for x in Path("./en_corpus/").glob("**/*.txt")]
tokenizer = BPE_token()
# train the tokenizer model
tokenizer.bpe_train(paths) 
# saving the tokenized data in our specified folder 
save_path = 'tokenized_data'
tokenizer.save_tokenizer(save_path)
