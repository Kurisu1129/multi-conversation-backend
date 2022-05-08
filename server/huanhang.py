

with open('./data/1.txt', 'r', encoding='utf-8') as f1, open('./data/2.txt', 'w', encoding='utf-8')as f2:
    str = ''
    for item in f1:
        str = str + item
    split = str.split('|')
    for item in split:
        f2.write(item + '\n')
