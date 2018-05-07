This is the code for Low-rank geometric mean metric learning. 

Pre-requisites:
MATLAB (the following gode for tested with MATLAB R2017a, but it should also run on previous versions without much issue).

Instructions:
0. Run the following in MATLAB.
1. run_me_first: This files adds appropriate folders/files to path. You should start with it.
2. runner: Runs the LR-GMML code for a dataset for various values of t, for a specific number of times. It gives back min, max and average of test errors and writes it to file. For the metrics shows in paper, we are taking average of 5 runs. 

One should be able to reproduce our results by running the above two steps. 

Note:
Step number 2 may take some time because we are running the algorithm lot's of times. One can limit the values of "t" and numRandomIterations to limit the number of runs.   