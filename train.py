import torch

import model
import tokenizer

tokens = tokenizer.sentence2idx('Hello World')
llm = model.MiniTransformer(vocab_size=96, embedding_dim=4)

output = llm(tokens)

