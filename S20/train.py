"""
Train our Tokenizers on some data, just to see them in action.
"""

import os
import time
from kannadaTokenizer import KannadaTokenizer

# open some text and train a vocab of 512 tokens
text = open("kannadatxt.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()

# construct the Tokenizer object and kick off verbose training
tokenizer = KannadaTokenizer()
tokenizer.train(text, 5024, verbose=True)
# writes two files in the models directory: name.model, and name.vocab
prefix = os.path.join("models", "kannada")
tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")