from datasets import load_dataset

dataset = load_dataset('roneneldan/TinyStories')

def filter(text):
    return text.replace('\n\n', '\n')

print(len(dataset['train']['text']))

stories = dataset['train']['text']

num_stories = len(stories)

with open('train.txt', 'w') as file:
    for i in range(num_stories):
        print(''.join([filter(stories[i]), '\n']), file=file)
