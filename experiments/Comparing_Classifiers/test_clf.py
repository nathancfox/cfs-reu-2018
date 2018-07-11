import numpy as np
import pprint as pp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def kfold_test(clf, k, data, target):
    cm = np.zeros((k, 2, 2))
    cnt = 0
    kf = StratifiedKFold(n_splits=k)
    for train_index, test_index in kf.split(data, target):
        clf.fit(data[train_index], target[train_index])
        y_pred = clf.predict(data[test_index])
        cm[cnt] = confusion_matrix(target[test_index], y_pred)
        cnt += 1
    accs = np.zeros(k)
    sens = np.zeros(k)
    spec = np.zeros(k)
    cnt = 0
    for m in cm:
        accs[cnt] = (m[0,0]+m[1,1])/m.sum()
        sens[cnt] = m[1,1]/(m[1,1]+m[1,0])
        spec[cnt] = m[0,0]/(m[0,0]+m[0,1])
        cnt += 1
    accuracy = accs.mean()
    sensitivity = sens.mean()
    specificity = spec.mean()
    return ((accuracy, sensitivity, specificity), cm)

data = np.loadtxt('data/data.csv', delimiter=',')
target = np.loadtxt('data/target.csv', delimiter=',')

with open('results.txt', 'w') as f:
    classifiers = [SVC, KNeighborsClassifier, GaussianNB, RandomForestClassifier]
    for c in classifiers:
        clf = c()
        scores, cm = kfold_test(clf, 5, data, target)
        f.write('{}\n'.format(c) + '-'*80 + '\n')
        f.write('\n')
        f.write('    Accuracy    : {}\n'.format(scores[0]))
        f.write('    Sensitivity : {}\n'.format(scores[1]))
        f.write('    Specificity : {}\n'.format(scores[2]))
        f.write('\n')
        f.write('    Confusion Matrices:\n\n')
        for line in str(cm).splitlines():
            f.write('        '+line+'\n')
        f.write('\n'+'='*80+'\n\n')
