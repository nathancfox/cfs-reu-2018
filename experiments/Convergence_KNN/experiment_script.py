import os

project = 'reu'
queue = 'general'
runtime = '10:00'
processors = '4'
email = 'nathanfox@miami.edu'

for i in range(3):
    filename = '{:02}_report_spread'.format(i)
    os.system('mkdir {}'.format(filename))
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
              + '--expname "Swarm Convergence - KNN: {:02}" '.format(i)
              + '--author "Nathan Fox" --outpath {} '.format(filename)
              + '--copyscript --initpart\n')
    os.system('bsub < {}/job_script_{:02}'.format(filename, i))
    # os.system('chmod +x {}/job_script'.format(filename))
    # os.system('{}/job_script > {}/output.txt 2> {}/error.txt &'.format(filename, filename, filename))
