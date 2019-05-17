import os
import sys
import numpy as np

project = 'reu'
queue = 'general'
runtime = '10:00'
processors = '4'
email = 'nathanfox@miami.edu'

for i in range(100):
    filename = '{:02}_iteration'.format(i)
    try:
        os.mkdir('{}'.format(filename))
    except FileExistsError:
        answer = input('Directory {}/ Exists. Overwrite? (y/n): '.format(filename))
        if answer.lower()[0] == 'y':
            os.system('rm {}/*'.format(filename))
        else:
            print('Exiting...')
            sys.exit(1)
    with open(filename+'/job_script_{:02}'.format(i), 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#BSUB -J {}\n'.format(filename))
        f.write('#BSUB -P {}\n'.format(project))
        f.write('#BSUB -o {}/output.txt\n'.format(filename))
        f.write('#BSUB -e {}/error.txt\n'.format(filename))
        f.write('#BSUB -W {}\n'.format(runtime))
        f.write('#BSUB -q {}\n'.format(queue))
        f.write('#BSUB -n {}\n'.format(processors))
        f.write('#BSUB -B\n')
        f.write('#BSUB -N\n')
        f.write('#BSUB -u {}\n'.format(email))
        f.write('\n')
        f.write('python cpso.py --npart 100 --ndim 190 --constants 2.1 2.1 2.1 '
              + '--terms accuracy sensitivity low_number '
              + '--weights 0.3 0.5 0.2 --xbounds -6.0 6.0 '
              + '--vbounds -3.0 1.5 --wbounds 0.4 0.9 --time 200 --gbestlimit 3 '
              + '--data data/data.csv --target data/target.csv '
              + '--labels data/feature_labels.csv '
              + '--expname "100 Runs of KNN {:02}" '.format(i)
              + '--author "Nathan Fox" --outpath {} '.format(filename)
              + '--copyscript --initpart\n')
    os.system('bsub < {}/job_script_{:02}'.format(filename, i))
    # os.system('chmod 764 {}/job_script_{:02}'.format(filename, i))
    # os.system('{}/job_script_{:02} &'.format(filename, i))
    # print('Running Job {:02}'.format(i))

project = 'reu'
queue = 'general'
runtime = '0:30'
processors = '1'
email = 'nathanfox@miami.edu'
  
# "Control"
for i in range(100):
    filename = 'control_{:02}_iteration'.format(i)
    try:
        os.mkdir('{}'.format(filename))
    except FileExistsError:
        answer = input('Directory {}/ Exists. Overwrite? (y/n): '.format(filename))
        if answer.lower()[0] == 'y':
            os.system('rm {}/*'.format(filename))
        else:
            print('Exiting...')
            sys.exit(1)
    with open(filename+'/job_script_{:02}'.format(i), 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#BSUB -J {}\n'.format(filename))
        f.write('#BSUB -P {}\n'.format(project))
        f.write('#BSUB -o {}/output.txt\n'.format(filename))
        f.write('#BSUB -e {}/error.txt\n'.format(filename))
        f.write('#BSUB -W {}\n'.format(runtime))
        f.write('#BSUB -q {}\n'.format(queue))
        f.write('#BSUB -n {}\n'.format(processors))
        f.write('#BSUB -B\n')
        f.write('#BSUB -N\n')
        f.write('#BSUB -u {}\n'.format(email))
        f.write('\n')
        f.write('python control_script.py --outpath {} '.format(filename)
              + '--data data/data.csv --target data/target.csv '
              + '--iter {} --author "Nathan Fox"\n'.format(i))
    os.system('bsub < {}/job_script_{:02}'.format(filename, i))
    # os.system('chmod 764 {}/job_script_{:02}'.format(filename, i))
    # os.system('{}/job_script_{:02} &'.format(filename, i))
    # print('Running Control {:02}'.format(i))
