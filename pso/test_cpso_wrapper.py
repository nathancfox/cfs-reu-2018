import time
start = time.time()
import sys
import os
import datetime
import argparse
import pickle
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
    return (int(h), int(m), int(s))
            
parser = argparse.ArgumentParser(description=('Run a single instance of'+
                                    ' the COMB-PSO feature selection'+
                                    ' algorithm.'))
parser.add_argument('--npart', type=int, required=True,
                    help='Number of particles in the swarm')
parser.add_argument('--ndim', type=int, required=True,
                    help='Number of features in the feature data')
parser.add_argument('-c', '--constants', nargs=3, type=float, dest='c',
                    required=True,
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
parser.add_argument('-t', '--time', type=int, dest = 't', required=True,
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
# Leftover from debugging
# for a in parser._actions:
#     print('\n'+str(a)+'\n')
# sys.exit(0)

args = parser.parse_args()

# Argument Checking

err_flag = False
if args.npart <= 0:
    err_flag = True
    print('Error: --npart must be greater than 0')
if args.ndim <= 0:
    err_flag = True
    print('Error: --ndim must be greater than 0')
if args.alpha < 0.0 or args.alpha > 1.0:
    err_flag = True
    print('Error: -a, --alpha must be in [0.0, 1.0]')
if args.test_size <= 0.0 or args.test_size >= 1.0:
    err_flag = True
    print('Error: --testsize must be in (0.0, 1.0)')
if args.x_bounds[0] >= args.x_bounds[1]:
    err_flag = True
    print('Error: --xbounds; first argument must',
          'be smaller than second argument')
if args.v_bounds[0] >= args.v_bounds[1]:
    err_flag = True
    print('Error: --vbounds; first argument must',
          'be smaller than second argument')
if args.w_bounds[0] >= args.w_bounds[1]:
    err_flag = True
    print('Error: --wbounds; first argument must',
          'be smaller than second argument')
if args.t <= 0:
    err_flag = True
    print('Error: -t, --time must be greater than 0')
if not os.path.isfile(args.data_path):
    err_flag = True
    print('Error: --data must be a valid csv file')
if not os.path.isfile(args.target_path):
    err_flag = True
    print('Error: --target must be a valid csv file')
if args.feature_labels:
    if not os.path.isfile(args.feature_labels):
        err_flag = True
        print('Error: --labels must be a valid csv file')
    args.feature_labels = np.loadtxt(args.feature_labels, delimiter=',',
                                     dtype=str)
else:
    args.feature_labels = [i for i in range(args.ndim)]    
if not os.path.isdir(args.output_path):
    err_flag = True
    print('Error: --outpath must be a valid directory')

if err_flag:
    sys.exit(1)

# Processing input into tuples because they are immutable.
args.x_bounds = tuple(args.x_bounds)
args.v_bounds = tuple(args.v_bounds)
args.w_bounds = tuple(args.w_bounds)
args.t_bounds = (0, args.t)

if args.output_path[-1] != '/':
    args.output_path.append('/')
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

s = COMB_Swarm(args.npart, args.c[0], args.c[1], args.c[2], args.ndim,
               args.alpha, args.test_size,
               args.x_bounds, args.v_bounds, args.w_bounds, args.t_bounds,
               args.data_path, args.target_path)
s.initialize_particles()
s.execute_search()
s.final_eval()
data = []
columns = []
for k, v in s.var_by_time.items():
    columns.append(k)
    data.append(v)
var_by_time = pd.DataFrame(data=np.transpose(data), columns=columns)
var_by_time.to_csv(args.output_path+'var_by_time.csv')
np.savetxt(args.output_path+'abinary.csv', s.abinary, delimiter=',')
np.savetxt(args.output_path+'X_train.csv', s.X_train, delimiter=',')
np.savetxt(args.output_path+'y_train.csv', s.y_train, delimiter=',')
np.savetxt(args.output_path+'X_test.csv', s.X_test, delimiter=',')
np.savetxt(args.output_path+'y_test.csv', s.y_test, delimiter=',')

with open(args.output_path+'pickled_trained_classifier', 'wb') as classifier:
    s.clf.fit(s.X_train[:, s.abinary], s.y_train)
    pickle.dump(s.clf, classifier)

end = time.time()

with open(args.output_path+'summary_results.out', 'a') as f:
    f.write('\n\n')
    f.write('#'*80+'\n')
    f.write('#\n')
    f.write('# {}\n'.format(args.expname))
    f.write('# {}\n'.format(args.author))
    dt = datetime.datetime.today()
    f.write('# {}/{}/{} {}:{}\n'.format(dt.month, dt.day, dt.year,
                                    dt.hour, dt.minute))
    f.write('#\n')
    f.write('#'*80+'\n')
    f.write('\n\n')
    f.write('Parameters\n----------\n')
    f.write('  {:40} : {:d}\n'.format('Number of Particles (npart)', args.npart))
    f.write('  {:40} : {:.2f}\n'.format('Acceleration Constant 1 (c1)', args.c[0]))
    f.write('  {:40} : {:.2f}\n'.format('Acceleration Constant 2 (c2)', args.c[1]))
    f.write('  {:40} : {:.2f}\n'.format('Acceleration Constant 3 (c3)', args.c[2]))
    f.write('  {:40} : {:d}\n'.format('Available Features (ndim)', args.ndim))
    f.write('  {:40} : {:.2f}\n'.format('Fitness Weight Constant (alpha)',
                                    args.alpha))
    f.write('  {:40} : {:.2f}\n'.format('Data Fraction Used As Test (test_size)',
                                    args.test_size))
    f.write('  {:40} : {}\n'.format('Position Bounds (x_bounds)',
                                str(args.x_bounds)))
    f.write('  {:40} : {}\n'.format('Velocity Bounds (v_bounds)',
                                str(args.v_bounds)))
    f.write('  {:40} : {}\n'.format('Inertia Bounds (w_bounds)',
                                str(args.w_bounds)))
    f.write('  {:40} : {:d}\n'.format('End Time (t_bounds[1])',
                                  args.t_bounds[1]))
    f.write('\n')
    f.write('  feature data : {}\n'.format(args.data_path))
    f.write('  target data  : {}\n'.format(args.target_path))
    f.write('\n')
    f.write('Results\n-------\n')
    f.write('\n')
    f.write('  Archived Best\n  =============\n')
    f.write('  {:40} : {:d}\n'.format('Number of Features',
                                  np.count_nonzero(s.abinary)))
    f.write('  {:40} : {:.4f}\n'.format('Fitness (a_fitness', s.a_fitness))
    f.write('  {:40} : {:.4f}\n'.format('Classifier Score on TRAINING DATA',
                                        s.a_score))
    f.write('  {:40} : {:.4f}\n'.format('Classifier Score on TEST DATA',
                                        s.final_scores.mean()))
    f.write('\n')
    f.write('  10-Fold Cross Validation Scores:\n')
    f.write('\n')
    for i in range(len(s.final_scores)):
        f.write('    {}\n'.format(s.final_scores[i]))
    f.write('\n')
    f.write('  Selected Features:\n\n')
    for i in range(len(args.feature_labels)):
        if s.abinary[i]:
            f.write('    {}\n'.format(args.feature_labels[i]))
    f.write('\n')
    runtime = seconds_readable(end-start)
    f.write('  {:40} : {:d}h {:d}m {:d}s\n'.format('Runtime', runtime[0],
                                               runtime[1], runtime[2]))
    f.write('\n\n')
    f.write('Output Files\n------------\n')
    f.write('  {:30} : '.format('var_by_time.csv'))
    tempdesc = ('This is a csv file that holds the values of several variables'
              + ' as they vary over time during a single run of the algorithm.'
              + ' Each row represents a single time, each column is a single'
              + ' variable. The first row is a header row and contains labels'
              + ' for all the columns except the first one, which is time.'
              + ' It is expected that the time column will be used as an index'
              + ' in any DataFrames used to process this file and so no'
              + ' column label was given.')
    tempdesc = tw.wrap(tempdesc, width=45)
    f.write(tempdesc[0]+'\n')
    for i in range(1, len(tempdesc)):
        f.write(' '*35 + tempdesc[i]+'\n')
    f.write('\n')
    f.write('  {:30} : '.format('abinary.csv'))
    tempdesc = ('This is a csv file that holds the boolean vector returned as'
              + ' the optimum feature subset by the algorithm. It is a list'
              + ' of values where each value is either a 0 or 1. In the order'
              + ' given in the original data set or in the label vector, if'
              + ' the value is a 0, the feature is excluded. If the value is'
              + ' a 1, the feature is included.')
    tempdesc = tw.wrap(tempdesc, width=45)
    f.write(tempdesc[0]+'\n')
    for i in range(1, len(tempdesc)):
        f.write(' '*35 + tempdesc[i]+'\n')
    f.write('\n')
    f.write('  {:30} : '.format('pickled_trained_classifier'))
    tempdesc = ('This is the trained classifier using the subset, abinary, of'
              + ' the randomly selected training data subsetted from the full'
              + ' feature data.')
    tempdesc = tw.wrap(tempdesc, width=45)
    f.write(tempdesc[0]+'\n')
    for i in range(1, len(tempdesc)):
        f.write(' '*35 + tempdesc[i]+'\n')
    f.write('\n')
    f.write('  {:30} : '.format('X_train.csv'))
    tempdesc = ('This is the subset of the feature data provided that was'
              + ' used as training data for the algorithm.')
    tempdesc = tw.wrap(tempdesc, width=45)
    f.write(tempdesc[0]+'\n')
    for i in range(1, len(tempdesc)):
        f.write(' '*35 + tempdesc[i]+'\n')
    f.write('\n')
    f.write('  {:30} : '.format('y_train.csv'))
    tempdesc = ('This is the subset of the target data provided that was'
              + ' used as training data for the algorithm.')
    tempdesc = tw.wrap(tempdesc, width=45)
    f.write(tempdesc[0]+'\n')
    for i in range(1, len(tempdesc)):
        f.write(' '*35 + tempdesc[i]+'\n')
    f.write('\n')
    f.write('  {:30} : '.format('X_test.csv'))
    tempdesc = ('This is the subset of the feature data provided that was'
              + ' used as testing data for the algorithm.')
    tempdesc = tw.wrap(tempdesc, width=45)
    f.write(tempdesc[0]+'\n')
    for i in range(1, len(tempdesc)):
        f.write(' '*35 + tempdesc[i]+'\n')
    f.write('\n')
    f.write('  {:30} : '.format('y_test.csv'))
    tempdesc = ('This is the subset of the target data provided that was'
              + ' used as testing data for the algorithm.')
    tempdesc = tw.wrap(tempdesc, width=45)
    f.write(tempdesc[0]+'\n')
    for i in range(1, len(tempdesc)):
        f.write(' '*35 + tempdesc[i]+'\n')
    f.write('\n')
    f.write('-'*80+'\n')
    f.write('  End of Experiment\n')
    f.write('-'*80+'\n')
