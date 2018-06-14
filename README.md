# Computing for Structure REU - Summer 2018

**Author:** Nathan Fox \<nathanfox@miami.edu\>

**Initialized:** June 1, 2018

**Mentors:**

    * Dr. Stefan Wuchty
    * Dr. Vance Lemmon
    * Hassen Dhrif

This project explores the use of a modified Particle Swarm Optimization (PSO)
algorithm called COMB-PSO (Combined Continuous and Binary Particle Swarm
Optimization) on kinase inhibitor data. COMB-PSO is designed for feature
selection to use as input for a machine learning algorithm. In this case,
the feature subset selected by COMB-PSO will be fed to a classifer that labels
a kinase inhibitor as neurite-outgrowth inducing or not neurite-outgrowth
inducing, based on phenotypic assay data and in vitro kinase inhibition profiles.

### Branches
**master:** No real development here. Represents stable work.

**cpso\_dev:** Development of the Python implementation of the COMB-PSO algorithm.

**eda:** Exploratory analysis of the kinase inhibitor data.

**exp\_design:** Designing wrapper scripts and reporting functionality to run
                 experiments with the particle/swarm classes.

The main differences are all found in the pso/ folder.
