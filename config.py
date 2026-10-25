
import torch


IS_DEBUG = True

N_BLOCKS = 6
N_HEADS = 4

EMBED_DIM = 16

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

if IS_DEBUG:
    print(f'Using device: {DEVICE}')
