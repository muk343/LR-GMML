function [minError, average_error, maxError] = getLRGMMLError(inputDataset, r, t, algorithm, numRandomIterations)

    fprintf('values: r = %d, t = %d, algo = %s, dataset = %s\n', r, t, algorithm, inputDataset)
         
    dataset = char(inputDataset);    
    load(dataset);
    
    cumulativeError = 0;
    minError = 100000;
    maxError = -111111;
    for loopNumber = 1:numRandomIterations
        rng(loopNumber+1)
        
        %Making constraints and choosing pairs:
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
         
         configurationParams.maxiter = 250;
         configurationParams.algo = algorithm;
         
         [U, B] = lowrank_metric_learning(d, r, t, S, D, configurationParams);
                
         B = (B+B')/2;

         newXtrain = U*(sqrtm(B)*(U'*Xtrain));
         newXtest = U*(sqrtm(B)*(U'*Xtest));
         
         Mdl = fitcknn(newXtrain',Ytrain,'NumNeighbors',5);
         trainClass = predict(Mdl,newXtrain');
         trainError = 1-sum(trainClass==Ytrain)/length(trainClass);
         testClass = predict(Mdl,newXtest');
         testError = 1-sum(testClass==Ytest)/length(testClass);
         
         fprintf('testError: %e\n',testError);
         
         if (maxError < testError)
             maxError = testError;
         end
         
         if (minError > testError) 
             minError = testError;
         end
         cumulativeError = cumulativeError + testError;
    end
    
    average_error = cumulativeError / numRandomIterations;
    
end

