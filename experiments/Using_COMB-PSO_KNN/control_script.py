import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import argparse
import sys
import os
import datetime

def kfold_test(clf, k, b, data, target):
    cm = np.zeros((k, 2, 2))
    cnt = 0
    kf = StratifiedKFold(n_splits=k)
    for train_index, test_index in kf.split(data, target):
        clf.fit(data[train_index][:, b], target[train_index])
        y_pred = clf.predict(data[test_index][:, b])
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

def main():
    parser = argparse.ArgumentParser(description=('10-fold CV on full feature '
                                                + 'dataset for control'))
    parser.add_argument('--outpath', dest='output_path', required=True,
                        help='Directory to store output files')
    parser.add_argument('--data', dest='data_path', required=True,
                        help='csv file containing the feature data; no labels;'
                           + ' row=data point, column=feature')
    parser.add_argument('--target', dest='target_path', required=True,
                        help='csv file containing the target classifications')
    parser.add_argument('--iter', dest='iteration', required=True, type=int,
                        help='Iteration number of the control')
    parser.add_argument('--author', default='Unknown Author',
                        help='Author Name (for report file); use quotes')

    args = parser.parse_args()

    err_flag = False
    if not os.path.isdir(args.output_path):
        err_flag = True
        print('Error: --outpath must be a valid directory')
    if not os.path.isfile(args.data_path):
        err_flag = True
        print('Error: --data must be a valid csv file')
    if not os.path.isfile(args.target_path):
        err_flag = True
        print('Error: --target must be a valid csv file')
    if args.iteration < 0:
        err_flag = True
        print('Error: --iter must be >= 0')
    
    if err_flag:
        sys.exit(1)

    # Downstream file saving code expects directories to end with a '/'
    if args.output_path[-1] != '/':
        args.output_path += '/'
    
    clf = KNeighborsClassifier()
    data = np.loadtxt(args.data_path, delimiter=',')
    target = np.loadtxt(args.target_path, delimiter=',')
    b = np.array([1 for i in range(data.shape[1])]).astype(bool)

    scores = kfold_test(clf, 10, b, data, target)

    with open(args.output_path+'summary_results.out', 'a') as f:
        f.write('\n\n')
        f.write('#'*80+'\n')
        f.write('#\n')
        f.write('# Using COMB-PSO - First Run: '
              + 'Control Iteration {}\n'.format(args.iteration))
        f.write('# {}\n'.format(args.author))
        dt = datetime.datetime.today()
        f.write('# {}/{}/{} {}:{}\n'.format(dt.month, dt.day, dt.year,
                                        dt.hour, dt.minute))
        f.write('#\n')
        f.write('#'*80+'\n')
        f.write('\n\n')
        f.write('Results:\n')
        f.write('----------------\n')
        f.write('\n')
        f.write('{:20} : {}\n'.format('Accuracy', scores[0]))
        f.write('{:20} : {}\n'.format('Sensitivity', scores[1]))
        f.write('{:20} : {}\n'.format('Specificity', scores[2]))


if __name__ == '__main__':
    main()
