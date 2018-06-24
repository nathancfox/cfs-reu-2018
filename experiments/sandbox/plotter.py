import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('number', type=int)
ns = parser.parse_args()

df = pd.read_csv('output/var_by_time.csv', index_col=0)
plt.style.use('ggplot')
fig = plt.figure(figsize=(20, 15))
numax = fig.add_subplot(211)
numax.plot(df.index, df['num_features'], 'ro')
scoreax = fig.add_subplot(212)
scoreax.plot(df.index, df['g_fitness'], 'o')
scoreax.plot(df.index, df['g_score'], 'o')
scoreax.plot(df.index, df['a_fitness'], 'o')
scoreax.plot(df.index, df['a_score'], 'o')
scoreax.legend()
fig.savefig('figures/plot_{}.svg'.format(ns.number))
