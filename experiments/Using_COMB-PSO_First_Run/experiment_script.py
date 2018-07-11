import os
import numpy as np

project = 'reu'
queue = 'general'
runtime = '1:30'
processors = '1'
email = 'nathanfox@miami.edu'

for i in range(10):
    filename = '{:02}_iteration'.format(i)
    os.system('mkdir {}'.format(filename))
    with open(filename+'/job_script', 'w') as f:
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
              + '--weights 0.1 0.7 0.2 --xbounds -6.0 6.0 '
              + '--vbounds -3.0 1.5 --wbounds 0.4 0.9 --time 200 --gbestlimit 3 '
              + '--data data/data.csv --target data/target.csv '
              + '--labels data/feature_labels.csv '
              + '--expname "Using COMB-PSO - First Run: Iteration {:02}" '.format(i)
              + '--author "Nathan Fox" --outpath {} '.format(filename)
              + '--copyscript --initpart\n')
    os.system('bsub < {}/job_script'.format(filename))
    # os.system('chmod 764 {}/job_script'.format(filename))
    # os.system('{}/job_script'.format(filename))

project = 'reu'
queue = 'general'
runtime = '0:20'
processors = '1'
email = 'nathanfox@miami.edu'

# "Control"
for i in range(3):
    filename = 'control_{:02}_iteration'.format(i)
    os.system('mkdir {}'.format(filename))
    with open(filename+'/job_script', 'w') as f:
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
    os.system('bsub < {}/job_script'.format(filename))
    # os.system('chmod 764 {}/job_script'.format(filename))
    # os.system('{}/job_script'.format(filename))
