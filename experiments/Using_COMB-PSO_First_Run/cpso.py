#------------------------------------------------------------------------------#
#
# Title: COMB-PSO Single Run Wrapper
# Author: Nathan Fox <nathanfox@miami.edu>
# Date Written: June 14, 2018
# Date Modified: July 5, 2018
#
#------------------------------------------------------------------------------#

"""Runs a single instance of the COMB-PSO feature selection algorithm.

This is a wrapper script, designed to be usable by non-programmers to access
the COMB_Particle and COMB_Swarm classes to run a single instance of the
COMB-Particle Swarm Optimization feature selection algorithm. It takes all
algorithm parameters as command line options and data as file arguments.
It outputs raw data as well as a detailed report in summary_report.out.

Example call:

    python3 cpso.py --npart 100 --ndim 2000 -c 2.0 2.0 2.0
    --terms accuracy low_number --weights 0.8 0.2
    --xbounds -6.0 6.0 --vbounds -4.0 1.0 --wbounds 0.4 0.9 -t 50 --gbestlimit 3
    --data path/to/feature/data.csv --target path/to/target/data.csv
    --labels path/to/feature/labels.csv --expname Test_Experiment_1
    --author "Nathan Fox" --outpath myoutputdirectory/ --copyscript
    --initpart

See Usage:

    python3 cpso.py -h

Most of the command line arguments are required. Only --gbestlimit, --labels,
--expname, --author, --outpath, --copyscript, and --initpart are optional.
"""

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
            
# Collect and Process Command Line Arguments
parser = argparse.ArgumentParser(description=('Run a single instance of'+
                                    ' the COMB-PSO feature selection'+
                                    ' algorithm.'))
parser.add_argument('--npart', type=int, required=True,
                    help='Number of particles in the swarm')
parser.add_argument('--ndim', type=int, required=True,
                    help='Number of features in the feature data')
parser.add_argument('-c', '--constants', nargs=3, type=float, dest='c',
                    required=True,
                    help=('3 arguments, in order: c1, c2, c3 '
                        + 'for velocity equation'))
parser.add_argument('--terms', nargs='+', required=True,
                    help=('Terms to be used in the fitness function. '
                        + 'Must be as many arguments as --weights. '
                        + 'Look in eval_fitness() in cpso_swarm.py '
                        + 'for options.'))
parser.add_argument('--weights', nargs='+', type=float, required=True,
                    help=('Weights to be used in the fitness function. Must '
                        + 'be as many arguments as --terms'))
parser.add_argument('--xbounds', nargs=2, type=float, dest='x_bounds',
                    required=True, help='2 arguments, min/max for position')
parser.add_argument('--vbounds', nargs=2, type=float, dest='v_bounds',
                    required=True, help='2 arguments, min/max for velocity')
parser.add_argument('--wbounds', nargs=2, type=float, dest='w_bounds',
                    required=True, help='2 arguments, min/max for inertia')
parser.add_argument('-t', '--time', type=int, dest = 't', required=True,
                    help='End time for algorithm')
parser.add_argument('--gbestlimit', type=int, dest='stagn_limit', default=3,
                    help=('Number of times gbest can stagnate in a row '
                        + 'before gbest is shuffled.'))
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
                    help='Author Name (for report file); use quotes')
parser.add_argument('--outpath', dest='output_path', default='./',
                    help='Directory to store output files')
parser.add_argument('--copyscript', action='store_true',
                    help='Make a copy of this script in --outpath')
parser.add_argument('--initpart', dest='init_particles', action='store_true',
                    help='Initialize particles automatically.')

args = parser.parse_args()

# Manual Argument Checking
err_flag = False
if args.npart <= 0:
    err_flag = True
    print('Error: --npart must be greater than 0')
if args.ndim <= 0:
    err_flag = True
    print('Error: --ndim must be greater than 0')
if len(args.terms) != len(args.weights):
    err_flag = True
    print('Error: --terms and --weights must have the same number of arguments')
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
if args.stagn_limit < 1:
    err_flag = True
    print('Error: --gbestlimit must be greater than 0')
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

# Processing input into tuples.
args.x_bounds = tuple(args.x_bounds)
args.v_bounds = tuple(args.v_bounds)
args.w_bounds = tuple(args.w_bounds)
args.t_bounds = (0, args.t)

# Creating the fitness function terms dict
terms = {}
for k, v in zip(args.terms, args.weights):
    terms[k] = v

# Downstream file saving code expects directories to end with a '/'
if args.output_path[-1] != '/':
    args.output_path += '/'

if args.copyscript:
    import shutil
    shutil.copy(__file__, args.output_path)

# Initialize swarm and execute algorithm
s = COMB_Swarm(args.npart, args.c[0], args.c[1], args.c[2], args.ndim, terms,
               args.stagn_limit, args.x_bounds, args.v_bounds, args.w_bounds,
               args.t_bounds, args.data_path, args.target_path,
               args.init_particles)
if not args.init_particles:
    print('TRIP')
    s.initialize_particles()
s.execute_search()
# Collect and process data from inside the swarm
data = []
columns = []
for k, v in s.var_by_time.items():
    if k == 'g_score':
        columns.append('g_accuracy')
        data.append([i[0] for i in v])
        columns.append('g_sensitivity')
        data.append([i[1] for i in v])
        columns.append('g_specificity')
        data.append([i[2] for i in v])
    elif k == 'a_score':
        columns.append('a_accuracy')
        data.append([i[0] for i in v])
        columns.append('a_sensitivity')
        data.append([i[1] for i in v])
        columns.append('a_specificity')
        data.append([i[2] for i in v])
    else:
        columns.append(k)
        data.append(v)
var_by_time = pd.DataFrame(data=np.transpose(data), columns=columns)
var_by_time.to_csv(args.output_path+'var_by_time.csv')
np.savetxt(args.output_path+'abinary.csv', s.abinary, delimiter=',')

end = time.time()

# Write summary_report.out
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
    f.write('  {:40} : {{\n'.format('Fitness Function Terms and Weights'))
    for k, v in terms.items():
        f.write(' '*47+'{:13}: {}\n'.format(k, v))
    f.write(' '*45+'}\n')
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
    f.write('  {:40} : {:.4f}\n'.format('Accuracy', s.a_score[0]))
    f.write('  {:40} : {:.4f}\n'.format('Sensitivity', s.a_score[1]))
    f.write('  {:40} : {:.4f}\n'.format('Specificity', s.a_score[2]))
    f.write('\n')
    f.write('  {:40} : '.format('Null Subsets')
          + '{} / '.format(s.all_false_counter)
          + '{} (Occurrences/Total Checks)\n'.format(
                s.npart * s.t_bounds[1]))
    f.write('\n')
    f.write('Final Re-Check 10-Fold Cross Validation Confusion Matrices:\n')
    f.write('\n')
    k = 10
    print(s.init_flag)
    print(s.gbinary)
    print(s.abinary)
    rechecked = s.test_classify(s.abinary, k)
    f.write(str(rechecked))
    f.write('\n\n')
    accs = np.zeros(k)
    sens = np.zeros(k)
    spec = np.zeros(k)
    cnt = 0
    for m in rechecked:
        accs[cnt] = (m[0,0]+m[1,1])/m.sum()
        sens[cnt] = m[1,1]/(m[1,1]+m[1,0])
        spec[cnt] = m[0,0]/(m[0,0]+m[0,1])
        cnt += 1
    f.write('    {:^13} | {:^13} | {:^13}\n'.format(
                'Accuracy', 'Sensitivity', 'Specificity'))
    f.write('    '+'-'*45+'\n')
    for i in range(len(rechecked)):
        f.write('    {:^13.4f} | {:^13.4f} | {:^13.4f}\n'.format(
                    accs[i], sens[i], spec[i]))
    f.write('    '+'-'*45+'\n')
    f.write('    '+'{:^45}\n'.format('Means'))
    f.write('    '+'-'*45+'\n')
    f.write('    {:^13.4f} | {:^13.4f} | {:^13.4f}\n'.format(
                accs.mean(), sens.mean(), spec.mean()))
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
    if args.copyscript:
        f.write('  {:30} : '.format('cpso_script.py'))
        tempdesc = ('This is the version of cpso.py that was used to generate '
                + 'this particular run of the algorithm.')
        tempdesc = tw.wrap(tempdesc, width=45)
        f.write(tempdesc[0]+'\n')
        for i in range(1, len(tempdesc)):
            f.write(' '*35 + tempdesc[i]+'\n')
    f.write('\n')
    f.write('-'*80+'\n')
    f.write('|{:^78}|\n'.format('End of Experiment'))
    f.write('-'*80+'\n')
