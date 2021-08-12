import glob, re

cnt = 0
path = "/home/z/Desktop/"
for name in glob.glob(path + "*.log"):
    with open(name, 'r') as r, open(str(cnt)+".log", 'w') as w:
        for i in r.readlines():
            i = i[27:].split(" ")
            for j in i:
                if len(j) > 7: continue
                try : 
                    float(j)
                    write = w.write(j)
                except :
                    pass
    cnt += 1