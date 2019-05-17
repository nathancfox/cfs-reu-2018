from statistics import mean, stdev

acc = [75.1111, 79.3846, 85.564, 82.3111, 85.9777, 83.3386, 81.9413,79.7157, 82.0088, 82.2484]
sen = [57, 71.9524, 86.6667, 75.8095, 84.2857, 78.6667, 72.381, 78.0952, 72.6667, 66.9524]
spe = [80.7143, 82.8849, 84.4286, 84.3636, 86.619, 85.119, 86.9381, 81.7932, 84.7179, 86.2098]

print('Accuracy\n'+'-'*20)
print('Mean    : {}'.format(mean(acc)))
print('Std Dev : {}'.format(stdev(acc)))
print()
print('Sensitivity\n'+'-'*20)
print('Mean    : {}'.format(mean(sen)))
print('Std Dev : {}'.format(stdev(sen)))
print()
print('Specificity\n'+'-'*20)
print('Mean    : {}'.format(mean(spe)))
print('Std Dev : {}'.format(stdev(spe)))
print()
