import torch
import torch.nn as nn

import config

class SelfAttention(nn.Module):
    def __init__(self, embed_dim=config.EMBED_DIM):
        super().__init__()

        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, token_embeddings):
        Q = self.query(token_embeddings)
        K = self.key(token_embeddings)
        V = self.value(token_embeddings)

        a = torch.arange(Q.size(-2), device=Q.device).unsqueeze(1)
        b = torch.arange(Q.size(-2), device=Q.device)
        mask = torch.where(a >= b, 0, float('-inf'))

        scores = Q @ K.transpose(-2, -1)
        scores = scores / (Q.size(-1) ** 0.5)
        scores += mask

        attention = torch.softmax(scores, dim=-1)
        output = attention @ V
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim=config.EMBED_DIM, n_heads=config.N_HEADS):
        super().__init__()

        assert embedding_dim % n_heads == 0
        self.d_head = embedding_dim // n_heads

        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

        self.head_projections = nn.ModuleList([
            nn.Linear(embedding_dim, self.d_head)
            for _ in range(n_heads)
        ])
        self.heads = nn.ModuleList([
            SelfAttention(self.d_head)
            for _ in range(n_heads)
        ])


    def forward(self, token_embeddings):
        attention_list = []
        for i, head in enumerate(self.heads):
            attention_list.append(head(self.head_projections[i](token_embeddings)))
        full_attention = torch.cat(attention_list, dim=-1)
        return self.output_projection(full_attention)


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


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim=config.EMBED_DIM):
        super().__init__()

        self.attention = MultiHeadAttention(embedding_dim)
        self.mlp = MLP(embedding_dim)

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, embeddings):
        attention_output = embeddings + self.attention(self.ln1(embeddings))
        mlp_output = attention_output + self.mlp(self.ln2(attention_output))

        if config.IS_DEBUG:
            print('Embedding:', embeddings)
            print('Embedding shape:', embeddings.shape)
            print()
            print('Attention output:', attention_output)
            print('Attention shape:', attention_output.shape)
            print()
            print('MLP output:', mlp_output)
            print('MLP shape:', mlp_output.shape)

        return mlp_output


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=config.EMBED_DIM):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, embedding_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim)
            for _ in range(config.N_BLOCKS)
        ])

    def forward(self, tokens):
        embeddings = self.embedding_table(tokens)
        for block in self.blocks:
            embeddings = block(embeddings)

        return embeddings