import pprint as pp
import pickle
import io
import datetime
import random
import numpy as np
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def kfold_test(svm_clf, k, b, data, target):
    cm = np.zeros((k, 2, 2))
    cnt = 0
    kf = StratifiedKFold(n_splits=k)
    for train_index, test_index in kf.split(data, target):
        svm_clf.fit(data[train_index][:, b], target[train_index])
        y_pred = svm_clf.predict(data[test_index][:, b])
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
    return (accuracy, sensitivity, specificity)

def single_test(svm_clf, b, data, target):
    cm = np.zeros((2, 2))
    X_train, X_test, y_train, y_test = train_test_split(data[:, b],
                                                        target, 
                                                        test_size = 0.25,
                                                        stratify = target)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0,0]+cm[1,1])/cm.sum()
    sensitivity = cm[1,1]/(cm[1,1]+cm[1,0])
    specificity = cm[0,0]/(cm[0,0]+cm[0,1])
    return (accuracy, sensitivity, specificity)
   
def cohort3(linked_groups):
    kinases = set()
    for group in linked_groups:
        kinases.add(random.choice(group))
    if len(kinases) != len(linked_groups):
        print('Error: Returned Set.size != Number of Groups.'
            + 'Check for membership repeats.')
        return
    return kinases

def main():
    with open('linked_groups.csv', 'r') as f:
        linked_groups = []
        for line in f:
            linked_groups.append(line.strip().split(sep=','))
    kernels = ['linear', 'rbf', 'poly', 'sigmoid', 'precomputed']
    cohorts = [ lambda : { 'ROCK2', 'PIK3CD', 'PRKCG', 'PRKG1', 'PRKX', 'TNK2',
                           'RPS6KA4', 'CDK5', 'MAPKAPK3', 'MAPK14' },
                lambda : { 'TNK2', 'ROCK1', 'PIK3CB', 'PRKCA', 'CDK1',
                           'RPS6KA4', 'PRKG1', 'MAPKAPK3', 'PRKX', 'MAPK14',
                           'MAP4K4', 'MUSK', 'FGR', 'EGFR', 'FLT4' },
                cohort3
              ]
    data = np.loadtxt('data/data.csv', delimiter=',')
    target = np.loadtxt('data/target.csv', delimiter=',')
    labels = np.loadtxt('data/feature_labels.csv', dtype=str, delimiter=',')
    scores = {k: {'kfold': [], 'single': []} for k in kernels}
    svm_clf = svm.SVC()
    for k in kernels:
        scores[k]['kfold'].append((0, kfold_test(svm_clf, 10,
                                                 np.isin(labels, list(cohorts[0]())),
                                                 data, target)))
        scores[k]['kfold'].append((1, kfold_test(svm_clf, 10,
                                                 np.isin(labels, list(cohorts[1]())),
                                                 data, target)))
        scores[k]['kfold'].append((2, kfold_test(svm_clf, 10,
                                                 np.isin(labels, list(cohorts[2](linked_groups))),
                                                 data, target)))
        scores[k]['kfold'].append((2, kfold_test(svm_clf, 10,
                                                 np.isin(labels, list(cohorts[2](linked_groups))),
                                                 data, target)))
        scores[k]['kfold'].append((2, kfold_test(svm_clf, 10,
                                                 np.isin(labels, list(cohorts[2](linked_groups))),
                                                 data, target)))

        scores[k]['single'].append((0, single_test(svm_clf,
                                                   np.isin(labels, list(cohorts[0]())),
                                                   data, target)))
        scores[k]['single'].append((1, single_test(svm_clf,
                                                   np.isin(labels, list(cohorts[1]())),
                                                   data, target)))
        scores[k]['single'].append((2, single_test(svm_clf,
                                                   np.isin(labels, list(cohorts[2](linked_groups))),
                                                   data, target)))
        scores[k]['single'].append((2, single_test(svm_clf,
                                                   np.isin(labels, list(cohorts[2](linked_groups))),
                                                   data, target)))
        scores[k]['single'].append((2, single_test(svm_clf,
                                                   np.isin(labels, list(cohorts[2](linked_groups))),
                                                   data, target)))
    with open('scores.pickle', 'wb') as f:
        pickle.dump(scores, f)
    with open('summary_report.txt', 'w') as f:
        idt = '    '

        f.write('\n\n')
        f.write('#'*100+'\n')
        f.write('#\n')
        f.write('# {}\n'.format('Verifying Rational Polypharmacology'))
        f.write('# {}\n'.format('Nathan Fox'))
        dt = datetime.datetime.today()
        f.write('# {}/{}/{} {}:{}\n'.format(dt.month, dt.day, dt.year,
                                        dt.hour, dt.minute))
        f.write('#\n')
        f.write('#'*100+'\n')
        f.write('\n\n')
        f.write('15 Polypharmacologically Linked Groups of Kinases (linked_groups.csv)\n')
        f.write('---------------------------------------------------------------------\n')
        buf = io.StringIO()
        pp.pprint(linked_groups, stream=buf, indent=2, width=100)
        for line in buf.getvalue().splitlines():
            f.write(idt+line+'\n')
        buf.close()
        f.write('\n')
        f.write('Cohort 1\n')
        f.write('-------\n')
        buf = io.StringIO()
        pp.pprint(cohorts[0](), stream=buf, indent=2, width=100)
        for line in buf.getvalue().splitlines():
            f.write(idt+line+'\n')
        buf.close()
        f.write('\n')
        f.write('Cohort 2:\n')
        f.write('-------\n')
        buf = io.StringIO()
        pp.pprint(cohorts[1](), stream=buf, indent=2, width=100)
        for line in buf.getvalue().splitlines():
            f.write(idt+line+'\n')
        buf.close()
        f.write('\n')
        f.write('Cohort 3:\n')
        f.write('-------\n')
        f.write('Randomly generated stratified set of 15 from the 15 groups in linked_groups.csv.\n')
        f.write('\n')
        f.write('='*100 + '\n')
        f.write('||  Results\n')
        f.write('='*100 + '\n')
        f.write('\n')
        for k in kernels:
            f.write('Kernel \'{}\'\n'.format(k))
            f.write('-------------------\n')
            f.write('\n')
            f.write(idt+'Single Split\n')
            f.write(idt+'============\n')
            f.write('\n')
            for s in scores[k]['single']:
                f.write(idt*2+'Cohort {}:\n'.format(s[0]))
                f.write(idt*3+'Accuracy: {}\n'.format(s[1][0]))
                f.write(idt*3+'Sensitivity: {}\n'.format(s[1][1]))
                f.write(idt*3+'Specificity: {}\n'.format(s[1][2]))
                f.write('\n')
            f.write(idt+'10-Fold CV\n')
            f.write(idt+'============\n')
            f.write('\n')
            for s in scores[k]['kfold']:
                f.write(idt*2+'Cohort {}:\n'.format(s[0]))
                f.write(idt*3+'Accuracy    : {}\n'.format(s[1][0]))
                f.write(idt*3+'Sensitivity : {}\n'.format(s[1][1]))
                f.write(idt*3+'Specificity : {}\n'.format(s[1][2]))
                f.write('\n')
            f.write('\n')
    
if __name__ == '__main__':
    main()
