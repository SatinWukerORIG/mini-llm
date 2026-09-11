import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / (Q.size(-1) ** 0.5)

        attention = torch.softmax(scores, dim=-1)
        output = attention @ V
        return output

class MLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        return self.model(x)


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = SelfAttention(embedding_dim)
        self.mlp = MLP(embedding_dim)

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, tokens):
        x = self.embedding(tokens)
        print('Embedding:', x)
        x = x + self.attention(self.ln1(x))
        print('Attention output:', x)
        x = x + self.mlp(self.ln2(x))
        print('MLP output:', x)
        return x

