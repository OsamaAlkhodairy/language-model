import torch
import torch.nn as nn
from torch.nn import functional as F

from config import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()

        assert n_embed % n_head == 0

        self.attn_kqv = nn.Linear(n_embed, n_embed * 3)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embed, n_embed)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape

        keys, queries, values = self.attn_kqv(x).split(n_embed, dim=2)
        keys    = keys.view(B, T, n_head, C // n_head).transpose(1, 2)    # (B, n_heads, T, head_size)
        queries = queries.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, n_heads, T, head_size)
        values  = values.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, n_heads, T, head_size)

        W = queries @ keys.transpose(-2, -1) * keys.shape[-1]**-0.5
        W = W.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        W = F.softmax(W, dim=-1)

        out = W @ values # (B, n_heads, T, T) x (B, n_heads, T, head_size) -> (B, n_heads, T, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.dropout(self.proj(out))

        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.self_attention = CausalSelfAttention(n_embed, n_head)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.self_attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self, vocab_size):
        print('vocab_size from Language Model', vocab_size)

        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding = nn.Embedding(block_size, n_embed)
        self.tr_blocks = nn.Sequential(*[TransformerBlock(n_embed, n_head) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        # Weight tying: using the weight matrix from the token embedding layer as the output weight matrix
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, inp, targets=None):
        B, T = inp.shape
        
        token_emb = self.token_embedding(inp)
        pos_emb = self.pos_embedding(torch.arange(T, device=device))

        x = token_emb + pos_emb
        x = self.tr_blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, context, max_new_tokens):
        for i in range(max_new_tokens):
            context_cur = context[:, -block_size:]
            
            logits, loss = self(context_cur)

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            new_tokens = torch.multinomial(probs, 1)

            context = torch.cat([context, new_tokens], dim=-1)

        return context
