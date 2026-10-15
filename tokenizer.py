
import torch

def sentence2idx(sentence: str) -> torch.Tensor:
    result = []
    for c in sentence:
        result.append(ord(c) - 32)

    return torch.as_tensor(result)

def idx2word(idx: int) -> str:
    return chr(idx + 32)

