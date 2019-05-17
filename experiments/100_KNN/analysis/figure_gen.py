import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

scores = pd.read_csv('numerical_data.csv')
control_scores = pd.read_csv('control_scores.csv')

# MR-SVM
#
# Accuracy
# --------------------
# Mean    : 81.76013
# Std Dev : 3.152035723870457
# 
# Sensitivity
# --------------------
# Mean    : 74.44763
# Std Dev : 8.520044410428076
# 
# Specificity
# --------------------
# Mean    : 84.37884
# Std Dev : 2.0529732980890607

fig = plt.figure(figsize=(8, 8), dpi=300, facecolor='w')
ax = fig.add_subplot(111)
ax.set_title('Overall Predictive Scores', size=24)
ax.errorbar([1, 2, 3], [control_scores['accuracy'].mean(), scores['a_accuracy'].mean(), 0.8176],
                yerr = [control_scores['accuracy'].std(), scores['a_accuracy'].std(), 0.0315],
                linestyle='None', marker='o', mec='k', mfc='r', ms=10, ecolor='k',
                capsize=10)
ax.errorbar([1, 2, 3], [control_scores['sensitivity'].mean(), scores['a_sensitivity'].mean(), 0.7445],
            yerr = [control_scores['sensitivity'].std(), scores['a_sensitivity'].std(), 0.0852],
            linestyle='None', marker='o', mec='k', mfc='g', ms=10, ecolor='k',
            capsize=10)
ax.errorbar([1, 2, 3], [control_scores['specificity'].mean(), scores['a_specificity'].mean(), 0.8438],
            yerr = [control_scores['specificity'].std(), scores['a_specificity'].std(), 0.0205],
            linestyle='None', marker='o', mec='k', mfc='b', ms=10, ecolor='k',
            capsize=10)
# ax.plot([3], [0.82], linestyle='None', marker='o', mec='k', mfc='r', ms=10)
# ax.plot([3], [0.72], linestyle='None', marker='o', mec='k', mfc='g', ms=10)
# ax.plot([3], [0.86], linestyle='None', marker='o', mec='k', mfc='b', ms=10)
handles = [Line2D([0], [0], marker='o', color='w', mfc='r', mec='k'),
           Line2D([0], [0], marker='o', color='w', mfc='g', mec='k'),
           Line2D([0], [0], marker='o', color='w', mfc='b', mec='k')]
ax.legend(handles, ['Accuracy', 'Sensitivity', 'Specificity'], loc=(0.75, 0.25))
ax.set_xlim((0.5, 3.5))
ax.set_ylim((0.0, 1.0))
ax.set_ylabel('Score', labelpad=10, size=16)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Control', 'COMB-PSO', 'MR-SVM'], size=16)
ax.set_yticks([i/10 for i in range(11)])
ax.set_yticklabels(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5',
                    '0.6', '0.7', '0.8', '0.9', '1.0'], size=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ax.plot([1.05, 1.45], [control_scores['accuracy'].mean(), 0.79756],
#         linestyle='--', color='r', alpha=0.15, marker='')
# ax.text(1.5, 0.79, '*', color='k', size=19, ha='center', va='center')
# ax.plot([1.55, 1.95], [0.80244, scores['a_accuracy'].mean()],
#         linestyle='--', color='r', alpha=0.15, marker='')
# ax.plot([1.05, 1.45], [control_scores['sensitivity'].mean(), 0.49811],
#         linestyle='--', color='g', alpha=0.15, marker='')
# ax.text(1.5, 0.50, '*', color='k', size=19, ha='center', va='center')
# ax.plot([1.55, 1.95], [0.52589, scores['a_sensitivity'].mean()],
#         linestyle='--', color='g', alpha=0.15, marker='')
# ax.plot([1.05, 1.45], [control_scores['specificity'].mean(), 0.915],
#         linestyle='--', color='b', alpha=0.15, marker='')
# ax.text(1.5, 0.90, '*', color='k', size=19, ha='center', va='center')
# ax.plot([1.55, 1.95], [0.911, scores['a_specificity'].mean()],
#         linestyle='--', color='b', alpha=0.15, marker='')

plt.savefig('test_figures/scores.png', dpi=300, format='png', bbox_inches='tight')

# Accuracy                       : 0.7780384045584043, 0.822146552706553
# Sensitivity                    : 0.38728571428571423, 0.6371607142857141
# Specificity                    : 0.9309356725146197, 0.8945467836257311

fig = plt.figure(figsize=(4, 4), dpi=300, facecolor='w')
ax = fig.add_subplot(111)
ax.set_title('Number of Features Returned for 100 Runs')
ax.boxplot([scores['num_features'], [17.6, 15.8, 21.2, 21.8, 21.4, 22, 23, 20.4, 19.6, 19.8]], medianprops={'color': 'r'})
#ax.boxplot([17.6, 15.8, 21.2, 21.8, 21.4, 22, 23, 20.4, 19.6, 19.8])
ax.set_ylabel('Num. of Features', labelpad=8)
ax.set_ylim((0, 37))
ax.set_xticks([1, 2])
ax.set_xticklabels(['COMB-PSO', 'MR-SVM'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('test_figures/num.png', dpi=300, format='png', bbox_inches='tight')
