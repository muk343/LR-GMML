clear all
close all;
% clc
warning('off', 'MATLAB:nargchk:deprecated');

rng(5);

%availableDatasets = [string('wine'), string('breast-cancer-wisconsin'), string('australian'), string('iris')];
availableDatasets = [string('iris')];

for datasetIndex = 1:length(availableDatasets)
    dataset = char(availableDatasets(datasetIndex));
    
    load(dataset);

    %t = 0.01; 0.001; 0.001; 0.9; 0.01; 0.5; % The weight parameter between similarities and dissimilarities. Refer to Sra's paper.

    c = length(unique(Y));
    n = size(X,2); % number of instances
    d = size(X,1); % dimensions

    index = randperm(n);
    numTrain = ceil(n/2);
    numTest = n-numTrain;
    trainIdx = index(1:numTrain);
    testIdx = index((numTrain+1):end);
    Xtrain = X(:,trainIdx);
    Ytrain = Y(trainIdx);
    Xtest = X(:,testIdx);
    Ytest = Y(testIdx);
    if (length(unique(Ytrain))~=c) || (length(unique(Ytest))~=c)
        fprintf('All classes are not present in train or test data\n');
        keyboard;
    end

    maxConstraint = numTrain*(numTrain-1)/2;
    givenNumConstraint = 40*c*(c-1);
    chosenConstraints = randperm(maxConstraint,givenNumConstraint);
    % mapping constraints to pairs 
    q = floor(sqrt(8*(chosenConstraints-1) + 1)/2 + 3/2);
    p = chosenConstraints - (q-1).*(q-2)/2;

    S = zeros(d);
    D = zeros(d);
    ms = 0;
    md = 0;
    for k=1:givenNumConstraint
        i = p(k);
        j = q(k);
        if Ytrain(i)==Ytrain(j)
            % similarity constraint
            temp2 = Xtrain(:,i) - Xtrain(:,j);
            S = S + temp2*temp2';
            ms = ms + 1;
        else
            % dissimilarity constraint
            temp3 = Xtrain(:,i) - Xtrain(:,j);
            D = D + temp3*temp3';
            md = md + 1;
        end
    end

     S = S/(ms + md);
     D = D/(ms + md);


    % %the number of constraints
    % hk = length(unique([Ytrain]));
    % params.const_factor = 40; %the coefficient of the number of constraints
    % num_const = params.const_factor * (hk * (hk-1));
    %
    % [S, D] = ConstGen(Xtrain', Ytrain', num_const);



    % Call algorithm
    %configurationParams.algo = 'steepestDescent';

    %r = 3; 15; d; 3; 7; 6; 7;; d-1; % or any < d. 
    tArray = [0.0001, 0.01, 0.1, 0.5, 0.9];
    %rankArray = [30, 25];
    availableAlgos = [string('steepestDescent'), string('trustRegions'), string('conjugateGradient')];
    %availableAlgos = [string('steepestDescent'), string('trustRegions')];

    configurationParams.maxiter = 250;

    fileID = fopen(strcat(dataset, 'retry4_with5.txt'),'w');

    for rIndex = 1:min(40, d)
        for tIndex = 1:length(tArray)
            for algoIndex = 1:length(availableAlgos)
                r = rIndex;
                %r = rankArray(rIndex);
                configurationParams.algo = availableAlgos(algoIndex);
                fprintf('values: r = %d, t = %d, algo = %s\n', r, tArray(tIndex), availableAlgos(algoIndex))
                [U, B] = lowrank_metric_learning(d, r, tArray(tIndex), S, D, configurationParams);

                B = (B+B')/2;

                newXtrain = U*(sqrtm(B)*(U'*Xtrain));
                newXtest = U*(sqrtm(B)*(U'*Xtest));

                Mdl = fitcknn(newXtrain',Ytrain,'NumNeighbors',5);
                trainClass = predict(Mdl,newXtrain');
                trainError = 1-sum(trainClass==Ytrain)/length(trainClass);
                testClass = predict(Mdl,newXtest');
                % trainError1 = resubLoss(Mdl);
                testError = 1-sum(testClass==Ytest)/length(testClass);

                fprintf('trainError: %e, testError: %e\n',trainError,testError);
                fprintf(fileID, '%d\t%d\t%d\t%e\t%e\n', r, tArray(tIndex), algoIndex, trainError, testError);

            end
        end
    end

    fprintf('done processing dataset: %s\n', dataset)
    fclose(fileID);

end

