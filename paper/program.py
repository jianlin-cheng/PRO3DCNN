import ytl

file = ytl.writeFileToList('outline.md')
for line in file:
    if(len(line)!=0 and line[0]=='#'):
        if '###' in line:
            print(' '*6+line.replace("#", ""))
        elif '##' in line:
            print(' '*3+line.replace("#", ""))
        elif '#' in line:
            print(''+line.replace("#", ""))