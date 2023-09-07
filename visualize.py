import ast
import torch

losses = []
with open('losses.txt', 'r') as f:
    file_contents = f.read()
    losses = ast.literal_eval(file_contents)

print(len(losses))

import matplotlib.pyplot as plt


final_losses = torch.pow(10, torch.tensor(losses)).view(-1, 1).mean(dim=1).tolist()

plt.plot(final_losses)
plt.show()