from datasets import load_dataset
import string

dataset = load_dataset("wikipedia", "20220301.simple")

def filter_text(text):
    text = text.replace('\n\n', '\n')
    text = text.replace('\n ', '\n')
    return text

train_split = dataset["train"]

all_text = ''.join([''.join(['\n\n\n', train_split[i]["title"], '\n', train_split[i]["text"]]) for i in range(len(train_split["text"]))])

filtered_text = filter_text(all_text)

letters_to_keep = list(string.ascii_lowercase) + list(string.ascii_uppercase) \
        + ['\n', ' ', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '-',
           '+', '=', '[', ']', '{', '}', '|', ':', ';', '"', "'", '<', '>', ',', '.',
           '?', '/', '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

letters_to_keep = set(letters_to_keep)
print('letters_to_keep', len(letters_to_keep))
print(sorted(list(letters_to_keep)))

filtered_text = ''.join([c for c in filtered_text if c in letters_to_keep])

cnt = {}
for c in filtered_text:
    if c in cnt:
        cnt[c] += 1
    else:
        cnt[c] = 1
print(cnt)

with open("input.txt", "w") as file:
    print(filtered_text, file=file)