# Freezes
This folder holds code that I wanted to freeze at a specific point in
development. Each freeze consists of a single \*.zip file named
with the date that it was created in YYMMDD format. (e.g. January
10, 2015 = 150110)

## Log

Name (Date)|Description
-----------|:----------
180613     |cpso\_particle.py and cpso\_swarm.py hold the COMB\_Particle and COMB\_Swarm classes respectively. These two classes are designed to be used in a COMB-PSO algorithm implementation. They are the bare minimum, hold very little error catching functionality, and do not report anything or export any data. This is the lightest version of the implementation and has no wrapper scripts with it.
180615     |cpso\_particle.py and cpso\_swarm.py hold the COMB\_Particle and COMB\_Swarm classes respectively. These two classes are designed to be used in a COMB-PSO algorithm implementation. They track some variables internally over time so that they can be extracted later for reporting by a wrapper. cpso.py is a fully functional wrapper script designed to be used from the command line, taking all parameters and data as command line arguments. It writes a detailed summary report and outputs raw data in .csv files to the designated output directory.
180620     |cpso\_particle.py and cpso\_swarm.py hold classes for use in implementing a COMB-PSO feature selection algorithm. cpso.py wraps it in a single instance. experiment\_script.py is an example experiment that generates job scripts and sends jobs to Pegasus, dynamically running an experiment with a changing parameter. data\_extracter.sh is an example bash script that runs through the summary\report.out files and retrieves certain data points of interest.
180703     |cpso\_particle.py and cpso\_swarm.py hold classes for use in implementing a COMB-PSO feature selection algorithm. cpso.py wraps it in a single instance. example\_experiment\_script.py is an example experiment that generates job scripts and sends jobs to Pegasus, dynamically running an experiment with a changing parameter. example\_data\_extracter.sh is an example bash script that runs through the summary\report.out files and retrieves certain data points of interest. test\_script.sh is a basic shell script that is used for easy testing of the cpso.py program.
