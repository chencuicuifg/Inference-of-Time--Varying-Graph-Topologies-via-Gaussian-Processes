function [K, dK, D2] = covMaha(k, dk, hyp, x)
%COVMAHA Computes the Squared Exponential Covariance Matrix and Its Derivative Using Mahalanobis Distance
%
% Syntax:
%   [K, dK, D2] = covMaha(k, dk, hyp, x)
%
% Inputs:
%   k    - Covariance function handle
%          A function that takes squared Euclidean distances and returns the covariance values.
%   dk   - Derivative of the covariance function handle
%          A function that takes squared Euclidean distances and covariance values,
%          and returns the derivative of the covariance with respect to the squared distances.
%   hyp  - Hyperparameters vector
%          A numeric vector containing hyperparameters required for covariance computation.
%          Typically, the last element represents the log-scale factor, and the preceding
%          elements are for the base covariance function (`covSE`).
%   x    - Input data matrix (T x N)
%          T: Number of observations or time steps
%          N: Number of features or dimensions in the data
%
% Outputs:
%   K   - Squared Exponential Covariance Matrix (T x T)
%         The covariance matrix computed using the Squared Exponential (SE) kernel scaled by hyperparameters.
%   dK  - Function handle for the derivative of the covariance matrix with respect to hyperparameters
%         This handle accepts a matrix `Q` and returns the derivative vector `dhyp`.
%   D2  - Squared Euclidean Distance Matrix (T x T)
%         The matrix containing squared distances between each pair of observations.
%
% Description:
%   The `covMaha` function computes a scaled Squared Exponential (SE) covariance matrix based on
%   the provided hyperparameters and input data. 
%   See Also:
%   covSE, maha, maha_dirder

    %% Ensure Hyperparameters are a Column Vector
    hyp = hyp(:);
    
    %% Define Scaling Function and Its Derivative
    A = @(x) bsxfun(@times, x, exp(2 * hyp'));
    dAdhyp = @(dAdiag) 2 * A(dAdiag')';
    
    %% Compute Mahalanobis Distance and Its Derivative
    [D2, dmaha] = maha(x, A);         % Compute Mahalanobis distance
    K = k(D2);                         % Evaluate covariance using the kernel function
    
    %% Define Derivative Function Handle
    dK = @(Q) dirder(Q, K, dk, D2, dmaha, dAdhyp);
    
end

%% Sub-function: dirder
function [dhyp] = dirder(Q, K, dk, D2, dmaha, dAdhyp)
% DIRDER Computes the Derivative of the Covariance Matrix with Respect to Hyperparameters
%
% Syntax:
%   dhyp = dirder(Q, K, dk, D2, dmaha, dAdhyp)
%
% Inputs:
%   Q       - Matrix for derivative computation (T x T)
%   K       - Squared Exponential Covariance Matrix (T x T)
%   dk      - Derivative of the covariance function handle
%             A function that takes squared Euclidean distances and covariance values,
%             and returns the derivative of the covariance with respect to the squared distances.
%   D2      - Squared Euclidean Distance Matrix (T x T)
%   dmaha   - Function handle for the derivative of Mahalanobis distance
%             (defined within the `maha` sub-function)
%   dAdhyp  - Function handle for the derivative of the scaling matrix with respect to hyperparameters
%             (defined as 2 * A(dAdiag')' within `covMaha`)
%
% Outputs:
%   dhyp    - Derivative of the covariance matrix with respect to hyperparameters (vector)
%
% Description:
%   The `dirder` function computes the derivative of the scaled covariance matrix `K` with respect
%   to the hyperparameters. 

    %% Compute Intermediate Matrix R
    R = dk(D2, K) .* Q;
    
    %% Compute Derivative of Mahalanobis Distance
    [dAdiag] = dmaha(R);
    
    %% Compute Final Derivative with Respect to Hyperparameters
    dhyp = dAdhyp(dAdiag);
    
end

%% Sub-function: maha
function [D2, dmaha] = maha(x, A)
% MAHA Computes the Squared Mahalanobis Distance Matrix and Its Derivative
%
% Syntax:
%   [D2, dmaha] = maha(x, A)
%
% Inputs:
%   x    - Input data matrix (T x N)
%          T: Number of observations or time steps
%          N: Number of features or dimensions in the data
%   A    - Scaling function handle
%          A function that scales the input data `x` based on hyperparameters.
%
% Outputs:
%   D2    - Squared Mahalanobis Distance Matrix (T x T)
%          Contains the squared Mahalanobis distances between each pair of observations.
%   dmaha - Function handle for the derivative of the Mahalanobis distance
%          This handle accepts a matrix `Q` and returns the derivative vector `dAdiag`.
%
% Description:
%   The `maha` function computes the squared Mahalanobis distance matrix `D2` for the input data `x`
%   after scaling it with the provided function handle `A`. It also defines a function handle `dmaha`
%   to compute the derivative of `D2` with respect to a directional matrix `Q`.
%
%   The Mahalanobis distance is computed as:
%     D2(i, j) = ||A(x_i) - A(x_j)||^2
%
%   where `A` scales the input data based on hyperparameters.


    %% Compute Mean of Each Feature
    mu = mean(x, 1);
    
    %% Center the Data by Subtracting the Mean
    x = bsxfun(@minus, x, mu);
    
    %% Apply Scaling Function to Centered Data
    Ax = A(x); 
    sax = sum(x .* Ax, 2);                             

    %% Compute Squared Mahalanobis Distance Matrix
    D2 = max(bsxfun(@plus, sax, bsxfun(@minus, sax', 2 * x * Ax')), 0);     % Ensure non-negative distances
    D2 = (D2 + D2') / 2;  % Symmetrize the distance matrix
    
    %% Define Derivative Function Handle for Mahalanobis Distance
    dmaha = @(Q) maha_dirder(Q, x); 
    
end

%% Sub-function: maha_dirder
function [dAdiag] = maha_dirder(Q, x)
% MAHA_DERDER Computes the Directional Derivative of the Mahalanobis Distance
%
% Syntax:
%   dAdiag = maha_dirder(Q, x)
%
% Inputs:
%   Q   - Matrix for derivative computation (T x T)
%   x   - Centered input data matrix (T x N)
%         Centered by subtracting the mean from each feature.
%
% Outputs:
%   dAdiag - Directional derivative vector (N x 1)
%
% Description:
%   The `maha_dirder` function computes the derivative of the squared Mahalanobis distance
%   with respect to the length scale parameters.
% See Also:
%   maha

    %% Compute Sums of Q
    q2 = sum(Q, 2);                       % Sum over columns (T x 1)
    q1 = sum(Q, 1)';                      % Sum over rows (T x 1)
    
    %% Define Symmetrization Function
    sym = @(X) (X + X') / 2;
    
    %% Determine Calculation Method Based on Number of Features
    dAdense = size(x, 2) < 5;             % Use dense matrix operations if N < 5
    
    %% Compute Intermediate Matrix y
    y = bsxfun(@times, q1 + q2, x) - (Q + Q') * x;
    
    %% Compute Derivative Based on Calculation Method
    if dAdense         % Use dense matrix operations for small N
        dA = sym(x' * y);              % Compute symmetric derivative matrix
        dAdiag = diag(dA);              % Extract diagonal elements
    else               % Use matrix-vector multiplications for larger N
        dAdiag = sum(x .* y, 1)';      % Compute sum across observations for each feature
    end
    
end
