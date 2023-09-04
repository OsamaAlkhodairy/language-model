import random

def is_balanced(s):
    cnt = 0
    for c in s:
        if c == '(':
            cnt += 1
        elif c == ')':
            cnt -= 1
        if cnt < 0:
            return False
    return cnt == 0

# This function generates a random balanced bracket sequence
def generate_random(n):
    s = ''
    while len(s) < n:
        if is_balanced(s) or random.random() < 0.5:
            s += '('
        else:
            s += ')'
    
    while is_balanced(s) == False:
        s += ')'
    
    return s

# Now, let's create a dataset of 1000 random balanced bracket sequences
with open('balanced-brackets.txt', 'w') as f:
    for i in range(10000):
        n = random.randint(1, 25)
        s = generate_random(n)
        f.write(s + '\n')
