# BROKEN DUE TO UPDATED CLASSES
# This is a repeat of an experiment and the proof of
# concept can be found in Swarm Convergence. The updated
# classes did not invalidate this test.

import numpy as np
from cpso_swarm import COMB_Swarm
from cpso_particle import COMB_Particle

s = COMB_Swarm(3, 2.1, 2.1, 2.1, 5,
               0.8, 0.2,
               (-6.0, 6.0), (-4.0, 0.25), (0.4, 0.9), (0, 10),
               'data/data.csv', 'data/target.csv')
s.swarm.append(COMB_Particle(2.1, 2.1, 2.1, 5, (-6.0, 6.0), (-4.0, 0.25), (0.4, 0.9)))
s.swarm.append(COMB_Particle(2.1, 2.1, 2.1, 5, (-6.0, 6.0), (-4.0, 0.25), (0.4, 0.9)))
s.swarm.append(COMB_Particle(2.1, 2.1, 2.1, 5, (-6.0, 6.0), (-4.0, 0.25), (0.4, 0.9)))
s.swarm[0].x = np.array([1, 32, 8, 4, 13])
s.swarm[1].x = np.array([24, 3, 28, 3, 21])
s.swarm[2].x = np.array([12, 5, 24, 15, 31])
print('{:35} : {}'.format('Calculated Spread', s.calc_spread()))
print('{:35} : {}'.format('Correct Spread', 59.863405937339586))
