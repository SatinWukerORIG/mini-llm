import torch

import model


network = model.MiniTransformer(vocab_size=5, embedding_dim=4)

input_tokens = torch.tensor([1, 2, 3, 4, 3, 2])
output = network(input_tokens)
