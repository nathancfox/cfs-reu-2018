import numpy as np
import os

dirlist = os.listdir('..')
dirlist.sort()

for i in range(len(dirlist)):
    try:
        test = int(dirlist[i][:2])
    except:
        start = i
        break

stop = len(dirlist)
for i in range(start, stop):
    del dirlist[start]

print('{:40} : Nonzero : Size : Ratio'.format('Directory'))
print('-'*100)
for d in dirlist:
    try:
        a = np.loadtxt('../{}/y_train.csv'.format(d), delimiter=',')
        print('{:40} : {:7} : {:4} : {:.4f}'.format(d, np.count_nonzero(a), a.size, np.count_nonzero(a)/a.size))
    except:
        pass
