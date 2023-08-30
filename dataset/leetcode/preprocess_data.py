import pandas as pd
import re

file = pd.read_csv('leetcode_raw.csv')

banned_chars = ['×', 'Σ']

def prepare(statement):
    # filter out unwanted statements in the dataset
    if statement == 'SQL Schema':
        return None
    
    for b in banned_chars:
        if statement.find(b) != -1:
            return None
    
    # Removing examples from the problem statement
    idx = statement.find("Example 1")
    if idx != -1:
        statement = statement[:idx]
    # Removing newlines from the problem statement
    statement = statement.replace('\n', '')
    # Adjusting spaces after the end of each sentence
    statement = re.sub(r'\.\s*', '. ', statement)
    statement = statement[:-1]
    # Adding newline character to indicate end of the problem statement
    statement = statement + '\n'

    statement = statement.replace('(', '[')
    statement = statement.replace(')', ']')
    statement = statement.replace('{', '[')
    statement = statement.replace('}', ']')
    statement = statement.replace('‘', '')
    statement = statement.replace('’', '')
    statement = statement.replace('\u200b', '')
    statement = statement.replace('“', '')
    statement = statement.replace('”', '')
    statement = statement.replace('Σ', '')
    statement = statement.replace('↑', '')
    statement = statement.replace('\\', '')
    statement = statement.replace('_', '')
    statement = statement.replace('@', '')
    statement = statement.replace('⌊', '')
    statement = statement.replace('⌋', '')
    statement = statement.replace('\t', '')
    statement = statement.replace('`', '')
    statement = statement.replace('→', '')
    statement = statement.replace('!', '.')
    statement = statement.replace('̸', '.')
    statement = statement.replace('√', '')
    statement = statement.replace('∞', '')
    statement = statement.replace('$', '')
    statement = statement.replace('∪', '')
    statement = statement.replace('≤', '<=')
    statement = statement.replace('≥', '>=')
    statement = statement.replace('−', '-')
    statement = statement.replace('⟶', '')
    statement = statement.replace('#', '')
    statement = statement.replace('%', '')
    statement = statement.replace('&', '')

    statement = statement.lower()

    return statement


problem_statements = file.loc[:]["description"]

prepared_statements = [prepare(statement) for statement in problem_statements]
prepared_statements = list(filter(lambda statement: statement != None, prepared_statements))

print(sorted(list(set(''.join(prepared_statements)))))

with open("leetcode_prepared.txt", "w") as file:
    for statement in prepared_statements:
        file.write(statement)