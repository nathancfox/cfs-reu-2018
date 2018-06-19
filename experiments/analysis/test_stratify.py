import numpy as np
from sklearn.model_selection import train_test_split

a = np.loadtxt('data.csv', delimiter=',')
b = np.loadtxt('target.csv', delimiter=',')
print()
print(' '*36 + 'Full Set')
print('='*80)
print()
print('{} | {} | {} | {:6}'.format('Index', 'Non-Zero', 'Size', 'Ratio'))
print('{:5} | {:8d} | {:4d} | {:6.4f}'.format(0, np.count_nonzero(b), b.size, np.count_nonzero(b) / b.size))
print()
print(' '*32 + 'stratify = None')
print('='*80)
print()
print('{} | {} | {} | {:6}'.format('Index', 'Non-Zero', 'Size', 'Ratio'))
print('-'*80)
for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(a, b, test_size=0.2)
    print('{:5} | {:8d} | {:4d} | {:6.4f}'.format(i, np.count_nonzero(y_train), y_train.size, np.count_nonzero(y_train) / y_train.size))
print()
print(' '*35 + 'stratify = target.csv')
print('='*80)
print()
print('{} | {} | {} | {:6}'.format('Index', 'Non-Zero', 'Size', 'Ratio'))
print('-'*80)
for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(a, b, test_size=0.2, stratify=b)
    print('{:5} | {:8d} | {:4d} | {:6.4f}'.format(i, np.count_nonzero(y_train), y_train.size, np.count_nonzero(y_train) / y_train.size))
