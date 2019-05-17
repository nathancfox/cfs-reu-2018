# Computing for Structure REU - Summer 2018

**Author:** Nathan Fox \<nchristopherfox@gmail.com\>

**Initialized:** June 1, 2018

### NOTE: This repository is unmaintained
This codebase was written for a one-time summer research project and is not
fully checked or proofed for active use. Some code documentation may be a
little out of date, and there may be a few bugs. If you are interested in
using any of this code, I highly recommend you contact me first.

**Mentors:**

    * Dr. Vance Lemmon
    * Dr. Stefan Wuchty
    * Hassen Dhrif

This project explores the use of a modified Particle Swarm Optimization (PSO)
algorithm called COMB-PSO (Combined Continuous and Binary Particle Swarm
Optimization) on kinase inhibitor data. COMB-PSO selects features that pipe to
a machine learning algorithm. In this case, the feature subset selected by COMB-PSO
will be fed to a classifier that labels a kinase inhibitor as neurite-outgrowth
inducing or not neurite-outgrowth inducing, based on phenotypic assay data and
in vitro kinase inhibition profiles.

### Branches
**master:** Actual experiments and analysis using stable scripts/implementation.

**eda:** Exploratory analysis of the kinase inhibitor data.

The master branch runs all experiments & analyses in the experiments/ folder.
The lab notebook can be found in experiments/lab_notebooks/cfs_notebook.md 
