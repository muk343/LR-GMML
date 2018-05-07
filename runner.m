clear all;
clc;

inputDataset = 'Isolet';
r = 45;
algorithm = 'conjugateGradient';
%algorithm = 'trustRegions';
numRandomIterations = 5;

tArray = [0.0001, 0.01, 0.1, 0.5, 0.9];
%tArray = [0.1, 0.5, 0.9];

fileID = fopen(strcat(inputDataset, 'averagedRunsCG_r45.txt'),'w');

for tIndex = 1:length(tArray)
    t = tArray(tIndex);
    [minError, avgError, maxError] = getLRGMMLError(inputDataset, r, t, algorithm, numRandomIterations);
    fprintf(fileID, 'algo: %s\tr = %d\tt = %d\tminError = %f\tavgError = %d\tmaxError = %d\n',algorithm, r, t, minError, avgError, maxError);
end
