import os

for i in range(3):
    filename = '{:02}_report_spread'.format(i)
    os.system('mkdir {}'.format(filename))
    with open(filename+'/job_script', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')
        f.write('function notifyme () {\n')
        f.write('\tstart=$(date +%s)\n')
        f.write('\t"$@"\n')
        f.write('\tpaplay ~/.local/sndfiles/ding_ding.wav\n')
        f.write('\tnotify-send "I\'m Finished!" "\\"$(echo $@)\\" took $(($(date +%s) - start)) seconds to finish."\n')
        f.write('}\n')
        f.write('\n')
        f.write('notifyme python cpso.py --npart 100 --ndim 190 --constants 2.1 2.1 2.1 '
              + '--alpha 0.8 --testsize 0.2 --xbounds -6.0 6.0 '
              + '--vbounds -4.0 0.25 '
              + '--wbounds 0.4 0.9 --time 150 --data data/data.csv '
              + '--target data/target.csv --labels data/feature_labels.csv '
              + '--expname "Swarm Convergence - {:02}" '.format(i)
              + '--author "Nathan Fox" --outpath {}/ '.format(filename)
              + '\n')
    os.system('chmod +x {}/job_script'.format(filename))
    os.system('{}/job_script > {}/output.txt 2> {}/error.txt &'.format(filename, filename, filename))
