import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64
block_size = 256
n_embed = 384
learning_rate = 3e-4
max_iters = 5000
eval_interval = 500
eval_iters = 200
dropout = 0.1
n_head = 6
n_layers = 6
# -------------------------

with open('input.txt', 'r') as f:
    text = f.read()

print(len(text))
print(text[:100])

chars = sorted(list(set(''.join(text))))
vocab_size = len(chars)
print('vocab_size', vocab_size)

stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for s, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', device)

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = {
        'train': train_data,
        'val': val_data}[split]

    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y

class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        
        self.linear_v = nn.Linear(n_embed, head_size)
        self.linear_k = nn.Linear(n_embed, head_size)
        self.linear_q = nn.Linear(n_embed, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, inp):
        B, T, C = inp.shape
        
        values = self.linear_v(inp)
        keys = self.linear_k(inp)
        queries = self.linear_q(inp)

        W = keys @ queries.transpose(-2, -1) * C**-0.5 # (batch_dim, time_dim, time_dim)
        W = W.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        W = F.softmax(W, dim=-1)

        out = W @ values

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
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
        head_size = n_embed // n_head
        self.mh_attention = MultiHeadAttention(n_head, head_size)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.ln1(self.mh_attention(x))
        x = x + self.ln2(self.ffwd(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding = nn.Embedding(block_size, n_embed)
        self.tr_blocks = nn.Sequential(*[TransformerBlock(n_embed, n_head) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

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


model = LanguageModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

model = model.to(device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

@torch.no_grad()
def eval_loss(split):
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        xb, yb = get_batch(split)

        logits, loss = model(xb, yb)
        losses[i] = loss.item()
    
    return losses.mean()

for steps in range(max_iters):
    if steps % eval_interval == 0:
        print(f"step {steps:4d}: train loss: {eval_loss('train'):.8f}, val_loss: {eval_loss('val'):.8f}")
    
    xb, yb = get_batch('train')
    
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print('val loss', eval_loss('val'))


print('generating')
context = torch.tensor([[0]], device=device)
context = model.generate(context, 200)
context = context.view(-1).tolist()
print(decode(context))