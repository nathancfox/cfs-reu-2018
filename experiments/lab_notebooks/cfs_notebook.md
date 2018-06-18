# Computing for Structure - Summer 2018 Lab Notebook
**Author:** Nathan Fox

**Date:** June 15, 2018 - ???

**Supervisors:** Vance Lemmon, Stefan Wuchty

## Table of Contents
Experiment|Date|Summary|Completed
:---------|----|:------|---------
[Tuning VBounds](#0001)|06/15/2018|Testing COMB-PSO on kinase inhibitor data with varying vbounds.|Yes
[Tuning VBounds Range](#0002)|06/16/2018|Testing COMB-PSO on kinase inhibitor data with a varying vbound range.|No
[Conceptual Brainstorming](#1001)|06/18/2018|Brainstorming and contemplating about the last two experiments.|No

----------------------------------------------------------------------------------------------------

## Tuning VBounds <a name="0001"></a>
June 15, 2018

### Question
How does changing the vbounds parameter in the COMB-PSO algorithm on the kinase inhibitor data
I was given change the accuracy and efficiency of the the algorithm?

### Hypothesis
Based on what Hassen Dhrif (author of COMB-PSO) told me in discussion, it seems likely that
changing the vbounds parameter might improve overfitting. I personally think it will also
impact the ability of the swarm to effectively explore the search space.

Hassen tells me that a vbounds like [-4.0, 0.25] helps the swarm prioritize higher accuracy
and fewer features. However, I don't understand why an asymmetrical (around 0) bounds
parameter will help that. It seems to me like it will just concentrate the swarm in a subset
of the search space. Analogy, if the search space was the cartesian plane where x ϵ [-10, 10]
and y ϵ [-10, 10] and the search space was centered on the origin, then an asymmetrical
velocity vector would result in heavy exploration of quadrant III over quadrant I. How do I
know that the search space is centered on (0, 0, ... , 0)?

### Experiment Design
I decided to keep a constant vbounds range and only move it. Essentially, I'm varying
the vbounds\_center while keeping vbounds\_range constant. I'm varying vbounds from
(-6.0, -2.0) to (2.0, 6.0) in increments of 0.2, i.e. {(-6.0, -2.0), (-5.8, -1.8), ... ,
(1.8, 5.8), (2.0, 6.0)}. For each of these, I am running a full COMB-PSO algorithm
over the kinase inhibitor data, attempting to identify features (kinases) that are
needed to correctly classify kinase inhibitors as {hit|non-hit}. 

Parameter|Value
:--------|-----
Number of Particles (npart)|100
Number of Features (ndim)|190
Acceleration Constants (c1, c2, c3)|2.1, 2.1, 2.1
Alpha (alpha)|0.8
Test Size (testsize)|0.2
X Bounds (x\_bounds)|(-6.0, 6.0)
V Bounds (v\_bounds)|INDEPENDENT VARIABLE
W Bounds (w\_bounds|(0.4, 0.9)
Time (t\_bounds[1])|300

#### Input
Feature Data: data/data.csv

Target Data: data/target.csv

Feature Labels: data/feature\_labels.csv

#### Output

Each iteration creates a directory called XX\_vbounds\_vv\_VV/ where XX = the iteration number,
vv = v minimum, and VV = v maximum. Inside each directory is a file list:

* abinary.csv
* cpso\_script.py
* error.txt
* output.txt
* job\_script
* pickled\_trained\_classifier
* summary\_results.out
* var\_by\_time.csv
* X\_train.csv
* y\_train.csv
* X\_test.csv
* y\_test.csv

pickled\_trained\_classifier and all \*.csv files are variable outputs at the end of the
algorithm. summary\_results.out is a comprehensive, auto-generated report. error.txt and
output.txt are stderr and stdout for the job. cpso\_script.py and job\_script are the
scripts used to run this experiment.

#### Running the Experiment

File List:

* cpso\_particle.py
* cpso\_swarm.py
* cpso.py
* experiment\_script.py
* data\_extracter.sh

Directory List:

* analysis/
* data/

cpso\_particle.py and cpso\_swarm.py respectively hold COMB\_Particle and COMB\_Swarm classes
used by cpso.py to run an experiment. The actual experiment code is called by job\_script
for each iteration from within experiment\_script.py which generates and runs all jobs.
data\_extracter.sh is a bash script that iterates over all summary\_report.out files and
extracts the important information. I then pipe the output to analysis/raw\_data.csv.

data/ holds the three data input files for the experiment. analysis/ holds anything processed
after the experiment was run: processed data, analysis notebooks, generated figures, etc.

This experiment was run with a series of dynamically created/called jobs on the Pegasus
supercomputer at the Center for Computational Sciences at the University of Miami by
user ncf30 in project reu. Output emails are dumped in the LSF-Pegasus folder in
\<nathanfox@miami.edu\>.

#### experiment\_script.py
```
import os
import numpy as np

project = 'reu'
queue = 'general'
runtime = '1:00'
processors = '1'
email = 'nathanfox@miami.edu'

counter = 0
for vmin in np.arange(-6.0, 2.1, 0.2):
    vmax = vmin + 4.0
    filename = '{:02}_vbounds_{:+.1f}_{:+.1f}'.format(counter, vmin, vmax)
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
              + '--alpha 0.8 --testsize 0.2 --xbounds -6.0 6.0 '
              + '--vbounds {:.1f} {:.1f} '.format(vmin, vmax)
              + '--wbounds 0.4 0.9 --time 300 --data data/data.csv '
              + '--target data/target.csv --labels data/feature_labels.csv '
              + '--expname "Tuning Velocity Bounds: v_bounds = '
              + '({:.1f}, {:.1f})" '.format(vmin, vmax)
              + '--author "Nathan Fox" --outpath {}/ '.format(filename)
	      + '--copyscript\n')
    os.system('bsub < {}/job_script'.format(filename))
    counter += 1
```

### Results/Analysis

#### Descriptive Statistics
|       | num\_of\_features | a\_fitness          | training\_accuracy  | test\_accuracy     | 
|-------|-------------------|---------------------|---------------------|--------------------| 
| count | 41.0              | 41.0                | 41.0                | 41.0               | 
| mean  | 39.4390243902439  | 0.7632048780487803  | 0.7558951219512198  | 0.7303146341463412 | 
| std   | 36.39783563653738 | 0.07145365613885518 | 0.04258470354023562 | 0.0526396835143203 | 
| min   | 2.0               | 0.675               | 0.6964              | 0.5983             | 
| 25%   | 2.0               | 0.6938              | 0.7159              | 0.6967             | 
| 50%   | 45.0              | 0.7254              | 0.7406              | 0.7367             | 
| 75%   | 77.0              | 0.8335              | 0.7948              | 0.7683             | 
| max   | 80.0              | 0.8571              | 0.824               | 0.8183             | 

#### Plots
##### Histograms of Numerical Dependent Variables

![IMAGE: Histogram - Number of Features](cfs_notebook_files/Tuning_VBounds_HIST_Num_of_Features.svg)
![IMAGE: Histogram - a\_fitness](cfs_notebook_files/Tuning_VBounds_HIST_a_fitness.svg)
![IMAGE: Histogram - Training Accuracy](cfs_notebook_files/Tuning_VBounds_HIST_a_score.svg)
![IMAGE: Histogram - Test Accuracy](cfs_notebook_files/Tuning_VBounds_HIST_test_a_score.svg)

##### Pair Plot of all Relevant Variables
![IMAGE: Histogram - Pair Plot](cfs_notebook_files/Tuning_VBounds_pairplot.svg)

##### Scatter Plots of Interest

![IMAGE: Scatter - vmin vs. Number of Features](cfs_notebook_files/Tuning_VBounds_SCAT_vmin_num.svg)
![IMAGE: Scatter - vmin vs. Fitness Score](cfs_notebook_files/Tuning_VBounds_SCAT_vmin_afit.svg)
![IMAGE: Scatter - vmin vs. Training Accuracy](cfs_notebook_files/Tuning_VBounds_SCAT_vmin_train_score.svg)
![IMAGE: Scatter - vmin vs. Test Accuracy](cfs_notebook_files/Tuning_VBounds_SCAT_vmin_test_score.svg)

The two most important plots are Figures 1 and 4. There is an extremely strong sigmoid relationship
between the location of vbounds (the center) and the number of features included in the final
result. I'm not sure why, maybe an artifact of the logistic function used in the continuous to
binary position converstion? This is probably what Hassen was talking about when he said that
a vbounds skewed on the negative side helped reduce the number of features selected. However, this
is somewhat riduculous, it only gives two options. Additionally, there is no apparent relationship
between the location of the vbounds center and the resulting TEST accuracy.

Something else to note, the sigmoid relationship in Figure 1 appears to be bleeding into Figures
2 and 3. Apparently, the number of features has an impact on a\_fitness and training accuracy.
The impact on a\_fitness is expected (the fitness function included a weight on number
of features where fewer is better), but I'm surprised that it stratified the testing accuracy.
Perhaps this is because the fitness function is used to determine velocity?

### Conclusions/Next Questions
Disappointingly, I saw no relationship between a moving window for vbounds and the ultimate
accuracy of the classifier on the test data. Surprisingly, I saw a strong sigmoid relationship
between the value for vbounds and the number of features in the returned optimum subset.
The logistic function is used to convert a continuous position vector into a binary position
vector. I wonder if this is an artifact of that function. It seems extremely inconvenient.
With the rest of the parameters, moving vbounds causes the algorithm to shift between
two drastically different values. Additionally, this appears to bleed into the results of
the classification accuracy on TRAINING data.

Next questions include whether or not I can tune the y offset of the sigmoid relationship
between vmin and number of features, or if necessary how to ameliorate it.

----------------------------------------------------------------------------------------------------

## Tuning VBounds - Range <a name="0002"></a>
June 16, 2018

### Question
How does change the size of the range of vbounds alter the accuracy of the classifier using the
returned subset? How does it affect other dependent variables like the fitness or number of
features selected? In addition, does the placement of the center change the effect of the
changing range?

### Hypothesis
Based on my previous experiment, [Tuning VBounds](#0001), I think that the placement of the center
will matter a great deal. I think that each set will have its own range of results. However, I
am not sure if it will affect the *relationship* that vrange has on the results. I hypothesize
that a larger range will return generally higher TRAINING scores, because a small velocity
prohibits exploration and limits the particles to exploitation behavior. Additionally, I
hypothesize that a small range limited to > 0 or < 0 will produce unusual results.

### Experiment Design
This experiment is divided into 5 cohorts, where each cohort has a different vbounds center.
For example, if vbounds = (-6.0, -2.0), the center = -4.0. I chose centers at {-4.0, -2.0,
0.0, 2.0, 4.0}. For each cohort, I varied the range of vbounds from 0.0 to 8.0, or +/- 0.0
to +/- 4.0. For each vbounds in each cohort, I ran a COMB-PSO algorithm to produce results.

I will examine the number of features returned, the fitness score, and the TRAINING/TEST
scores for the returned subset, based on both the center of the vbounds and the range
of the vbounds.

Parameter|Value
:--------|-----
Number of Particles (npart)|100
Number of Features (ndim)|190
Acceleration Constants (c1, c2, c3)|2.1, 2.1, 2.1
Alpha (alpha)|0.8
Test Size (testsize)|0.2
X Bounds (x\_bounds)|(-6.0, 6.0)
V Bounds (v\_bounds)|INDEPENDENT VARIABLE
V Range (v\_bounds[1] - v\_bounds[0])| INDEPENDENT VARIABLE
W Bounds (w\_bounds|(0.4, 0.9)
Time (t\_bounds[1])|300

#### Input
For each cohort...

* Feature Data: data/data.csv
* Target Data: data/target.csv
* Feature Labels: data/feature\_labels.csv

#### Output
Each cohort is contained in a separate directory inside the experiment directory. For each
cohort...

Each iteration creates a directory called XX\_vcenter\_VV\_vrange\_RR/ where XX = the
iteration number, VV = vcenter, and RR = vrange. Inside each directory is a file list:

* abinary.csv
* cpso\_script.py
* error.txt
* output.txt
* job\_script
* pickled\_trained\_classifier
* summary\_results.out
* var\_by\_time.csv
* X\_train.csv
* y\_train.csv
* X\_test.csv
* y\_test.csv

pickled\_trained\_classifier and all \*.csv files are variable outputs at the end of the
algorithm. summary\_results.out is a comprehensive, auto-generated report. error.txt and
output.txt are stderr and stdout for the job. cpso\_script.py and job\_script are the
scripts used to run this experiment.

#### Running the Experiment
For each cohort...

File List:

* cpso\_particle.py
* cpso\_swarm.py
* cpso.py
* experiment\_script.py
* data\_extracter.sh

Directory List:

* analysis/
* data/

cpso\_particle.py and cpso\_swarm.py respectively hold COMB\_Particle and COMB\_Swarm classes
used by cpso.py to run an experiment. The actual experiment code is called by job\_script
for each iteration from within experiment\_script.py which generates and runs all jobs.
data\_extracter.sh is a bash script that iterates over all summary\_report.out files and
extracts the important information. I then pipe the output to analysis/raw\_data.csv.

data/ holds the three data input files for the experiment. analysis/ holds anything processed
after the experiment was run: processed data, analysis notebooks, generated figures, etc.

This experiment was run with a series of dynamically created/called jobs on the Pegasus
supercomputer at the Center for Computational Sciences at the University of Miami by
user ncf30 in project reu. Output emails are dumped in the LSF-Pegasus folder in
\<nathanfox@miami.edu\>.


#### experiment\_script.py for Each Cohort
Center = -4.0
```
import os
import numpy as np

project = 'reu'
queue = 'general'
runtime = '2:00'
processors = '1'
email = 'nathanfox@miami.edu'

counter = 0
vcenter = -4.0
for vhalfrange in np.arange(0.0, 4.0, 0.2):
    vmax = vcenter + vhalfrange
    vmin = vcenter - vhalfrange
    filename = '{:02}_vcenter_-4.0_vrange_{:+.1f}'.format(counter, 2*vhalfrange)
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
              + '--alpha 0.8 --testsize 0.2 --xbounds -6.0 6.0 '
              + '--vbounds {:.1f} {:.1f} '.format(vmin, vmax)
              + '--wbounds 0.4 0.9 --time 300 --data data/data.csv '
              + '--target data/target.csv --labels data/feature_labels.csv '
              + '--expname "Tuning Velocity Bounds: v_bounds = '
              + '({:.1f}, {:.1f})" '.format(vmin, vmax)
              + '--author "Nathan Fox" --outpath {}/ '.format(filename)
	      + '--copyscript\n')
    os.system('bsub < {}/job_script'.format(filename))
    counter += 1
```

Center = -2.0
```
import os
import numpy as np

project = 'reu'
queue = 'general'
runtime = '2:00'
processors = '1'
email = 'nathanfox@miami.edu'

counter = 0
vcenter = -2.0
for vhalfrange in np.arange(0.0, 4.0, 0.2):
    vmax = vcenter + vhalfrange
    vmin = vcenter - vhalfrange
    filename = '{:02}_vcenter_-2.0_vrange_{:+.1f}'.format(counter, 2*vhalfrange)
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
              + '--alpha 0.8 --testsize 0.2 --xbounds -6.0 6.0 '
              + '--vbounds {:.1f} {:.1f} '.format(vmin, vmax)
              + '--wbounds 0.4 0.9 --time 300 --data data/data.csv '
              + '--target data/target.csv --labels data/feature_labels.csv '
              + '--expname "Tuning Velocity Bounds: v_bounds = '
              + '({:.1f}, {:.1f})" '.format(vmin, vmax)
              + '--author "Nathan Fox" --outpath {}/ '.format(filename)
	      + '--copyscript\n')
    os.system('bsub < {}/job_script'.format(filename))
    counter += 1
```

Center = 0.0
```
import os
import numpy as np

project = 'reu'
queue = 'general'
runtime = '2:00'
processors = '1'
email = 'nathanfox@miami.edu'

counter = 0
vcenter = 0.0
for vhalfrange in np.arange(0.0, 4.0, 0.2):
    vmax = vcenter + vhalfrange
    vmin = vcenter - vhalfrange
    filename = '{:02}_vcenter_0.0_vrange_{:+.1f}'.format(counter, 2*vhalfrange)
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
              + '--alpha 0.8 --testsize 0.2 --xbounds -6.0 6.0 '
              + '--vbounds {:.1f} {:.1f} '.format(vmin, vmax)
              + '--wbounds 0.4 0.9 --time 300 --data data/data.csv '
              + '--target data/target.csv --labels data/feature_labels.csv '
              + '--expname "Tuning Velocity Bounds: v_bounds = '
              + '({:.1f}, {:.1f})" '.format(vmin, vmax)
              + '--author "Nathan Fox" --outpath {}/ '.format(filename)
	      + '--copyscript\n')
    os.system('bsub < {}/job_script'.format(filename))
    counter += 1
```

Center = 2.0
```
import os
import numpy as np

project = 'reu'
queue = 'general'
runtime = '2:00'
processors = '1'
email = 'nathanfox@miami.edu'

counter = 0
vcenter = 2.0
for vhalfrange in np.arange(0.0, 4.0, 0.2):
    vmax = vcenter + vhalfrange
    vmin = vcenter - vhalfrange
    filename = '{:02}_vcenter_2.0_vrange_{:+.1f}'.format(counter, 2*vhalfrange)
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
              + '--alpha 0.8 --testsize 0.2 --xbounds -6.0 6.0 '
              + '--vbounds {:.1f} {:.1f} '.format(vmin, vmax)
              + '--wbounds 0.4 0.9 --time 300 --data data/data.csv '
              + '--target data/target.csv --labels data/feature_labels.csv '
              + '--expname "Tuning Velocity Bounds: v_bounds = '
              + '({:.1f}, {:.1f})" '.format(vmin, vmax)
              + '--author "Nathan Fox" --outpath {}/ '.format(filename)
	      + '--copyscript\n')
    os.system('bsub < {}/job_script'.format(filename))
    counter += 1
```

Center = 4.0
```
import os
import numpy as np

project = 'reu'
queue = 'general'
runtime = '2:00'
processors = '1'
email = 'nathanfox@miami.edu'

counter = 0
vcenter = 4.0
for vhalfrange in np.arange(0.0, 4.0, 0.2):
    vmax = vcenter + vhalfrange
    vmin = vcenter - vhalfrange
    filename = '{:02}_vcenter_4.0_vrange_{:+.1f}'.format(counter, 2*vhalfrange)
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
              + '--alpha 0.8 --testsize 0.2 --xbounds -6.0 6.0 '
              + '--vbounds {:.1f} {:.1f} '.format(vmin, vmax)
              + '--wbounds 0.4 0.9 --time 300 --data data/data.csv '
              + '--target data/target.csv --labels data/feature_labels.csv '
              + '--expname "Tuning Velocity Bounds: v_bounds = '
              + '({:.1f}, {:.1f})" '.format(vmin, vmax)
              + '--author "Nathan Fox" --outpath {}/ '.format(filename)
	      + '--copyscript\n')
    os.system('bsub < {}/job_script'.format(filename))
    counter += 1
```

### Results/Analysis
I made two mistakes in coding the experiment. I forgot that the `cpso.py` script will complain
and quit if v_bounds[1]-v_bounds[0] <= 0, so all of the vrange = 0.0 data points were useless
(1 per cohort). Additionally, in `experiment_script.py`, I forgot that the `np.linspace` method's
stop parameter is excluded. So I'm missing a data point from the end where vrange = 8.0. I don't
think that either of these mistakes invalidated my data, so I proceeded normally.

NOTE: Something about some of the cohorts is causing a negative correlation between TRAINING
score and TEST score. Explore later. Is it a third party like num_of_features?

### Conclusions/Next Questions

----------------------------------------------------------------------------------------------------

## Conceptual Brainstorming <a name="1001"></a>
June 18, 2018

Why did the num_of_features variable have a sigmoid relationship with the moving vbounds in
[Tuning VBounds](#0001)? Because the position vector represents the propensity for that feature
to be above/below the random value, or included vs. excluded. If x_i < 0, then S(x_i) < 0.5 and
the odds are better that the feature will be excluded. However, the logistic function doesn't
give an uniform chance. It gives a extremely drastic chance either one way or the other. Small
changes in position can drastically change the chance of inclusion for a feature. Thus, if the
velocity is skewed negatively, then the position is far more likely to be negative, thus the
number of features included is small.

However, is this desirable? Would it be better to replace the sigmoid function with a linear
function? Such that S(x_i) = x_i/(x_bounds[1]-x_bounds[0])? Then the relationship would be far
less extreme and the stochasticity would play a larger value? If I were to do this, it would mean
that the stochasticity would begin to play a true role instead of merely a perturbance. That might
mean that I would have to treat it more like a Monte Carlo simulation and run it multiple times.

If the sigmoid behavior is desired, how do I move the function? Do I want a range that results
in the middle of the curve? How do I move the upper and lower asymptotes?

Other thoughts about Hassan's previous approach of Maximum Relevance and Iterative Checking. What
order did he iteratively check each of the 50 kinases? Was it in the same order each time? A random
order each time? It is important to note that kinases do not exist independently, but tend to affect
each other. Thus, because Hassan is not checking all permutations, (2^190 permutations is
computationally impossible), the order in which he checks is biologically relevant. Could I improve
his results merely by running it multiple times and shuffling the order each time, then take
the highest?

Things to do:

* Get the list of 50 best kinases and run the iterative algorithm, but shuffling each time.
* Try changing the sigmoid function to a linear function.
* Try changing alpha to 1.0
* Try adapting the fitness function so that the second term is more intelligent. Fewer is good,
  but too few is not.
* Read up on the pathways known to affect neurite outgrowth.
* Take a given position vector and convert it to binary many times, evaluate all the outputs.
* Get the pharmacologically linked kinase families and look at which families are activated
  for the returned subsets. Is it always the same families? Always the same kinases in those
  families? If they're actually pharmacologically linked, then the algorithm should give back
  only a consistent subset of families, not a consistent subset of kinases. Because with regard
  to informational contribution, most kinases inside a pharmacologically linked family are
  approximately the same. Then, it seems likely that each family represents a single branch
  pathway. (doesn't matter which kinase you knock down in a single branch pathway). However,
  kinases that don't align perfectly with their family may be intersection kinases.

I want to test that last bullet point on a different dataset. I want a phenotype that is
extremely well characterized and has the full regulation network known. Thus, I can test for
my theory on single branch vs intersection nodes.

----------------------------------------------------------------------------------------------------

## TITLE <a name="0003"></a>
DATE

### Question

### Hypothesis

### Experiment Design


Parameter|Value
:--------|-----

#### Input

#### Output

#### Running the Experiment

File List:

Directory List:

#### EXPERIMENT SCRIPT
```
```

### Results/Analysis

### Conclusions/Next Questions

----------------------------------------------------------------------------------------------------
