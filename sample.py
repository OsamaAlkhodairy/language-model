import torch
from model import LanguageModel

state_dict = torch.load('model.pth')
print('state_dict', state_dict.keys())
print('shape', state_dict['tr_blocks.0.self_attention.attn_kqv.weight'].shape)

vocab_size = 65

model = LanguageModel(vocab_size)
model.load_state_dict(torch.load('model.pth'))

model.eval()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', device)

model.to(device)


print('generating')
context = torch.tensor([[0]], device=device)
context = model.generate(context, 2000)
context = context.view(-1).tolist()
print(context)
