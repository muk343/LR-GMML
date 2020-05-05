function[Uopt, Bopt] = lowrank_metric_learning(d,r,t,S,D, configurationParams)
    %     clear; clc;
    %     rng(10);
    
    %     d = 100; % Oriignal feature space.
    %     r = 5; % Reduced feature space.
    %     TS = 500; % Number of similar pairs.
    %     TD = 100; % Number of dissimilar pairs.
    %
    %     S = randn(d, TS); S = S*S';
    %     D = randn(d, TD); D = D*D';
    %
    %     S = S/max(TS, TD);
    %     D = D/max(TS, TD);
    
    
    
    %     t = 0.5; 0.5; 0.0001; 0.999;
    
    
    problem.M = grassmannfactory(d, r);
    PD = sympositivedefinitefactory(r);
    
    % Cost
    problem.cost = @cost;
    function [f, store] = cost(U, store)
        if ~isfield(store, 'B')
            store.Stilde = U'*S*U;
            store.Dtilde = U'*D*U;
            store.B = getB(store.Stilde, store.Dtilde);
        end
        Stilde = store.Stilde;
        Dtilde = store.Dtilde;
        B = store.B;
        
        f = (1-t)*(PD.dist(inv(Stilde), B))^2 ...
            + t*(PD.dist(Dtilde, B))^2;
        f = f/4;
    end
    
    % Euclidean gradient
    problem.egrad = @grad;
    function [g, store] = grad(U, store)
        if ~isfield(store, 'B')
            [~, store] = cost(U, store);
        end
        Stilde = store.Stilde;
        Dtilde = store.Dtilde;
        B = store.B;
        
        % PD.log(X, B) = X^0.5 logm(X^{-0.5}*B* X^{-0.5} ) X^0.5.
        
        % So the finaly formula turns out to be
        
        % (1-t).m_s*(PD.log(inv(Stilde), B))*ALS - t.m_d*(Dtilde\(PD.log(Dtilde,
        % B)/Dtilde)*ALD
        g = (1-t)*S*U*(PD.log(inv(Stilde), B))...
            - t*D*U*(Dtilde\(PD.log(Dtilde, B)/Dtilde));
    end
    
    function B = getB(myS, myD)
        L = sqrtm(myS);
        M = L*myD*L;
        %u and s are squre, u: eigenvectors and s: diagonal matrix of
        %eigenvalues
        [u, s] = eig(M);
        %taking out the absolute values eigen values in an array
        diags = abs(diag(s));
        
        %t is as mentioned in paper, take in as a separate arg.
        %element wise power t
        Mt = u*(diag(diags.^t))*u';
        
        % L^(-1) Mt L^(-1)
        %
        B = (L\Mt)/L;
    end
    
    %     % Check gradient
    %     checkgradient(problem);
    %     pause;
    
    %     problem.linesearch = @(x, u) 0.01;
    %     options.ls_max_steps = 0;

    % Call algorithm,
    options.maxiter = configurationParams.maxiter; % maxiter
    
    algoName = configurationParams.algo;
    if strcmp(algoName, 'steepestDescent')
        [Uopt, ~, infos] = steepestdescent(problem,[],options);
    elseif strcmp(algoName, 'trustRegions')
        [Uopt, ~, infos] = trustregions(problem,[],options);
    elseif strcmp(algoName, 'conjugateGradient')
        [Uopt, ~, infos] = conjugategradient(problem,[],options);
    else
        ME = MException('Algo not found %s not found', algoName);
    throw(ME)
    end
    
    
    
    Stildeopt = Uopt'*S*Uopt;
    Dtildeopt = Uopt'*D*Uopt;
    Bopt = getB(Stildeopt, Dtildeopt);
    
    objective = 0.25*((1-t)*(PD.dist(inv(Stildeopt), Bopt))^2 + t*PD.dist(Dtildeopt, Bopt)^2);
    fprintf('objective: %e\n',objective);
end

