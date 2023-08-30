import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

from model import LanguageModel

from config import *

with open('input.txt', 'r') as f:
    text = f.read()

print(len(text))
print(text[:100])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('device', device)


# Character based model
# chars = sorted(list(set(''.join(text))))
# vocab_size = len(chars)
# print('vocab_size', vocab_size)

# stoi = {s:i for i, s in enumerate(chars)}
# itos = {i:s for s, i in stoi.items()}

# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: ''.join(itos[i] for i in l)

# data = torch.tensor(encode(text), dtype=torch.long)

# token based model
enc = tiktoken.get_encoding("gpt2")

text = enc.encode_ordinary(text)
print('first 100 tokens', text[:100])

vocab_size = max(list(text)) + 1
print('vocab size is', vocab_size)

data = torch.tensor(text, dtype=torch.long)

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



model = LanguageModel(vocab_size)
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