import ast

losses = []
with open('losses.txt', 'r') as f:
    file_contents = f.read()
    losses = ast.literal_eval(file_contents)

print(len(losses))

import matplotlib.pyplot as plt

plt.plot(losses)
plt.show()