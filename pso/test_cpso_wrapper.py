import time as t
start = t.time()
from cpso_particle import COMB_Particle
from cpso_swarm import COMB_Swarm
import numpy as np
import pandas as pd

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

s = COMB_Swarm(npart, c1, c2, c3, ndim, alpha, test_size,
               x_bounds, v_bounds, w_bounds, t_bounds,
               data_path, target_path)
s.initialize_particles()
s.execute_search()
data = []
columns = []
for k, v in s.report.items():
    columns.append(k)
    data.append(v)
reported = pd.DataFrame(data=np.transpose(data), columns=columns)
reported.to_csv('./reported.csv')
end = t.time()
print('\n\nHey I\'m done!!!\nThat took {} seconds!\n'.format(end-start))
