ann = ['a', 'b', 'c', 'd', 'e']
with open('ai.txt', 'w') as f:
    for i,a in enumerate(ann):
        f.write(str(i)+' '+a + '\n')
