import time as t
start = t.time()
import sys
import os
import datetime
import argparse
import textwrap as tw
from cpso_particle import COMB_Particle
from cpso_swarm import COMB_Swarm
import numpy as np
import pandas as pd

def seconds_readable(seconds):
    """Converts an integer number of seconds to hours, minutes, and seconds.

    Parameters
    ----------
    seconds : integers; number of seconds to be converted.

    Returns
    -------
    converted : integer tuple, size 3; (h, m, s) where the arguments
                are the time given in seconds in hours, minutes, seconds.

    Raises
    ------
    None
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return (h, m, s)
            
parser = argparse.ArgumentParser(description=('Run a single instance of'+
                                    ' the COMB-PSO feature selection'+
                                    ' algorithm.'))
parser.add_argument('--npart', type=int, required=True,
                    help='Number of particles in the swarm')
parser.add_argument('--ndim', type=int, required=True,
                    help='Number of features in the feature data')
parser.add_argument('-c', '--constants', nargs=3, type=float, required=True,
                    help=('3 arguments, in order: c1, c2, c3'
                         + ' for velocity equation'))
parser.add_argument('-a', '--alpha', type=float, dest='alpha', required=True,
                    help='α in the fitness equation, weight factor')
parser.add_argument('--testsize', type=float, dest='test_size', required=True,
                    help='Fraction of feature data reserved for final testing')
parser.add_argument('--xbounds', nargs=2, type=float, dest='x_bounds',
                    required=True, help='2 arguments, min/max for position')
parser.add_argument('--vbounds', nargs=2, type=float, dest='v_bounds',
                    required=True, help='2 arguments, min/max for velocity')
parser.add_argument('--wbounds', nargs=2, type=float, dest='w_bounds',
                    required=True, help='2 arguments, min/max for inertia')
parser.add_argument('-t', '--time', type=int, required=True,
                    help='End time for algorithm')
parser.add_argument('--data', dest='data_path', required=True,
                    help='csv file containing the feature data; no labels;'
                       + ' row=data point, column=feature')
parser.add_argument('--target', dest='target_path', required=True,
                    help='csv file containing the target classifications')
parser.add_argument('--labels', dest='feature_labels', default=None,
                    help='csv file containing the feature labels')
parser.add_argument('--expname', default='Generic Experiment',
                    help='Experiment Name (for report file)')
parser.add_argument('--author', default='Unknown Author',
                    help='Author Name (for report file)')
parser.add_argument('--outpath', dest='output_path', default='./',
                    help='Directory to store output files')
parser.parse_args()

# Argument Checking

err_flag = False
if npart <= 0:
    err_flag = True
    print('Error: --npart must be greater than 0')
if ndim <= 0:
    err_flag = True
    print('Error: --ndim must be greater than 0')
if alpha < 0.0 or alpha > 1.0:
    err_flag = True
    print('Error: -a, --alpha must be in [0.0, 1.0]')
if test_size <= 0.0 or test_size >= 1.0:
    err_flag = True
    print('Error: --testsize must be in (0.0, 1.0)')
if x_bounds[0] >= x_bounds[1]:
    err_flag = True
    print('Error: --xbounds; first argument must',
          'be smaller than second argument')
if v_bounds[0] >= v_bounds[1]:
    err_flag = True
    print('Error: --vbounds; first argument must',
          'be smaller than second argument')
if w_bounds[0] >= w_bounds[1]:
    err_flag = True
    print('Error: --wbounds; first argument must',
          'be smaller than second argument')
if t <= 0:
    err_flag = True
    print('Error: -t, --time must be greater than 0')
if not os.path.isfile(data_path):
    err_flag = True
    print('Error: --data must be a valid csv file')
if not os.path.isfile(target_path):
    err_flag = True
    print('Error: --target must be a valid csv file')
if not os.path.isdir(output_path):
    err_flag = True
    print('Error: --outpath must be a valid directory')

if err_flag:
    sys.exit(1)

# Processing input into tuples because they are immutable.
x_bounds = tuple(x_bounds)
v_bounds = tuple(v_bounds)
w_bounds = tuple(w_bounds)
t_bounds = (0, t)

if not feature_labels:
    feature_labels = [i for i in range(ndim)]    

"""
# Defaults saved while I test the command line options.
npart = 50
c1 = 2.1
c2 = 2.1
c3 = 2.1
ndim = 190
alpha = 0.8
test_size = 0.3
x_bounds = (-6.0, 6.0)
v_bounds = (-2.0, 2.0)
w_bounds = (0.4, 0.9)
t_bounds = (0, 200)
data_path = './working_data/prepped_for_classifier/data.csv'
target_path = './working_data/prepped_for_classifier/target.csv'
"""

s = COMB_Swarm(npart, c[1], c[2], c[3], ndim, alpha, test_size,
               x_bounds, v_bounds, w_bounds, t_bounds,
               data_path, target_path)
s.initialize_particles()
s.execute_search()
data = []
columns = []
for k, v in s.report.items():
    columns.append(k)
    data.append(v)
var_by_time = pd.DataFrame(data=np.transpose(data), columns=columns)
var_by_time.to_csv(output_path+'var_by_time.csv')
np.savetxt(output_path+'abinary.csv', s.abinary, delimiter=',')
np.savetxt(output_path+'X_train.csv', s.X_train, delimiter=',')
np.savetxt(output_path+'y_train.csv', s.y_train, delimiter=',')
np.savetxt(output_path+'X_test.csv', s.X_test, delimiter=',')
np.savetxt(output_path+'y_test.csv', s.y_test, delimiter=',')

with open(output_path+'pickled_trained_classifier', w) as classifier:
    s.clf.fit(s.X_train[:, s.abinary])
    pickle.dump(s.clf, classifier)

end = t.time()

with open(output_path+'summary_results.out', a) as f:
    print('\n')
    print('#'*80)
    print('#')
    print('# {}'.format(expname))
    print('# {}'.format(author))
    dt = datetime.datetime.today()
    print('# {}/{}/{} {}:{}'.format(dt.month, dt.day, dt.year,
                                    dt.hour, dt.minute))
    print('#')
    print('#'*80)
    print('\n')
    print('Parameters\n----------')
    print('  {:40} : {:d}'.format('Number of Particles (npart)', npart))
    print('  {:40} : {:.2f}'.format('Acceleration Constant 1 (c1)', c1))
    print('  {:40} : {:.2f}'.format('Acceleration Constant 2 (c2)', c2))
    print('  {:40} : {:.2f}'.format('Acceleration Constant 3 (c3)', c3))
    print('  {:40} : {:d}'.format('Available Features (ndim)', ndim))
    print('  {:40} : {:.2f}'.format('Fitness Weight Constant (alpha)', alpha))
    print('  {:40} : {:.2f'.format('Data Fraction Used As Test (test_size)',
                                    test_size))
    print('  {:40} : {}'.format('Position Bounds (x_bounds)', str(x_bounds)))
    print('  {:40} : {}'.format('Velocity Bounds (v_bounds)', str(v_bounds)))
    print('  {:40} : {}'.format('Inertia Bounds (w_bounds)', str(w_bounds)))
    print('  {:40} : {:d}'.format('End Time (t_bounds[1])', t_bounds[1]))
    print()
    print('  feature data : {}'.format(data_path))
    print('  target data  : {}'.format(target_path))
    print('\n')
    print('Results\n-------')
    print('  Archived Best:\n')
    print('  {:40} : {:d}'.format('Number of Features',
                                  np.count_nonzero(s.abinary)))
    print('  {:40} : {:.4f}'.format('Fitness (a_fitness', s.a_fitness))
    print('  {:40} : {:.4f}'.format('Classifier Score (a_score)', s.a_score))
    print('  Selected Features:')
    for i in range(len(feature_labels)):
        if s.abinary[i]:
            print('    {}'.format(feature_labels[i]))
    print()
    runtime = seconds_readable(end-start)
    print('  {:40} : {:d}h {:d}m {:d}s'.format('Runtime', runtime[0],
                                               runtime[1], runtime[2]))
    print('\n')
    print('Output Files\n------------')
    print('  {:40} : '.format('var_by_time.csv'), end='')
    tempdesc = ('This is a csv file that holds the values of several variables'
              + ' as they vary over time during a single run of the algorithm.'
              + ' Each row represents a single time, each column is a single'
              + ' variable. The first row is a header row and contains labels'
              + ' for all the columns except the first one, which is time.'
              + ' It is expected that the time column will be used as an index'
              + ' in any DataFrames used to process this file and so no'
              + ' column label was given.')
    tempdesc = tw.wrap(tempdesc, width=35)
    print(tempdesc[0])
    for i in range(1, len(tempdesc)):
        print(' '*45 + tempdesc[i])
    print()
    print('  {:40} : '.format('abinary.csv'), end='')
    tempdesc = ('This is a csv file that holds the boolean vector returned as'
              + ' the optimum feature subset by the algorithm. It is a list'
              + ' of values where each value is either a 0 or 1. In the order'
              + ' given in the original data set or in the label vector, if'
              + ' the value is a 0, the feature is excluded. If the value is'
              + ' a 1, the feature is included.')
    tempdesc = tw.wrap(tempdesc, width=35)
    print(tempdesc[0])
    for i in range(1, len(tempdesc)):
        print(' '*45 + tempdesc[i])
    print('  {:40} : '.format('pickled_trained_classifier'), end='')
    tempdesc = ('This is the trained classifier using the subset, abinary, of'
              + ' the randomly selected training data subsetted from the full'
              + ' feature data.')
    tempdesc = tw.wrap(tempdesc, width=35)
    print(tempdesc[0])
    for i in range(1, len(tempdesc)):
        print(' '*45 + tempdesc[i])
    print()
    print('  {:40} : '.format('X_train.csv'), end='')
    tempdesc = ('This is the subset of the feature data provided that was'
              + ' used as training data for the algorithm.')
    tempdesc = tw.wrap(tempdesc, width=35)
    print(tempdesc[0])
    for i in range(1, len(tempdesc)):
        print(' '*45 + tempdesc[i])
    print()
    print('  {:40} : '.format('y_train.csv'), end='')
    tempdesc = ('This is the subset of the target data provided that was'
              + ' used as training data for the algorithm.')
    tempdesc = tw.wrap(tempdesc, width=35)
    print(tempdesc[0])
    for i in range(1, len(tempdesc)):
        print(' '*45 + tempdesc[i])
    print()
    print('  {:40} : '.format('X_test.csv'), end='')
    tempdesc = ('This is the subset of the feature data provided that was'
              + ' used as testing data for the algorithm.')
    tempdesc = tw.wrap(tempdesc, width=35)
    print(tempdesc[0])
    for i in range(1, len(tempdesc)):
        print(' '*45 + tempdesc[i])
    print()
    print('  {:40} : '.format('y_test.csv'), end='')
    tempdesc = ('This is the subset of the target data provided that was'
              + ' used as testing data for the algorithm.')
    tempdesc = tw.wrap(tempdesc, width=35)
    print(tempdesc[0])
    for i in range(1, len(tempdesc)):
        print(' '*45 + tempdesc[i])
    print()
    print('-'*80)
    print('  End of Experiment')
    print('-'*80)
