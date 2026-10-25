import torch
import torchviz

import config
import model
import tokenizer

tokens = tokenizer.sentence2idx('Hello World')
llm = model.MiniTransformer(vocab_size=96, embedding_dim=config.EMBED_DIM)

output = llm(tokens)

dot = torchviz.make_dot(output, params=dict(list(llm.named_parameters()) + [('input', tokens)]))

# 5. Save/Render the computation graph as an image
dot.format = 'png'
dot.render('torch_computation_graph')
