import os

project = 'reu'
queue = 'general'
runtime = '7:00'
processors = '4'
email = 'nathanfox@miami.edu'

for i in range(3):
    filename = '{:02}_report_velocities'.format(i)
    os.system('mkdir {}'.format(filename))
    with open(filename+'/job_script', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#BSUB -J {}'.format(filename))
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
              + '--alpha 0.8 --testsize 0.2 --xbounds -6.0 6.0 '
              + '--vbounds -4.0 0.25 '
              + '--wbounds 0.4 0.9 --time 150 --data data/data.csv '
              + '--target data/target.csv --labels data/feature_labels.csv '
              + '--expname "Mean Particle Velocity - {:02}" '.format(i)
              + '--author "Nathan Fox" --outpath {}/ '.format(filename)
              + '\n')
    os.system('bsub < {}/job_script'.format(filename))
