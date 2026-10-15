import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, token_embeddings):
        Q = self.query(token_embeddings)
        K = self.key(token_embeddings)
        V = self.value(token_embeddings)


        scores = Q @ K.transpose(-2, -1)
        a = torch.arange(Q.size(-2)).unsqueeze(1)
        b = torch.arange(Q.size(-2))
        mask = a >= b
        
        scores = scores / (Q.size(-1) ** 0.5)
        scores *= mask
        print(scores)

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
        self.embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.attention = SelfAttention(embedding_dim)
        self.mlp = MLP(embedding_dim)

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, tokens):
        embeddings = self.embedding_table(tokens)
        print('Embedding:', embeddings)
        attention_output = embeddings + self.attention(self.ln1(embeddings))
        print('Attention output:', attention_output)
        mlp_output = attention_output + self.mlp(self.ln2(attention_output))
        print('MLP output:', mlp_output)
        return mlp_output

