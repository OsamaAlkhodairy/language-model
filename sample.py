import torch
import tiktoken
from model import LanguageModel

model_path = 'model-download.pth'

state_dict = torch.load(model_path)
print('state_dict', state_dict.keys())
# print('shape', state_dict['tr_blocks.0.self_attention.attn_kqv.weight'].shape)


with open('dataset/tinystories/train.txt', 'r') as f:
    text = f.read()

print(len(text))
print(text[:100])
text = text[:100000000]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', device)

enc = tiktoken.get_encoding("gpt2")
text = enc.encode_ordinary(text)

vocab_size = max(list(text)) + 1
print('vocab size is', vocab_size)


model = LanguageModel(vocab_size)
model.load_state_dict(torch.load(model_path))

model.to(device)
model.eval()


print('generating')
context = torch.tensor([enc.encode('Once upon a time, there was a little girl named Lily.')], device=device)
context = model.generate(context, 2000)
context = context.view(-1).tolist()
print(enc.decode(context))

