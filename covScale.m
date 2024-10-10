function [K, dK] = covScale(hyp, x)
% COVSCALE Computes the Scaled Covariance Matrix and Its Derivative
%
% Syntax:
%   [K, dK] = covScale(hyp, x)
%
% Inputs:
%   hyp - Hyperparameters vector
%         A numeric vector containing all hyperparameters required for the covariance computation.
%         - The last element of `hyp` represents the log-scale factor (lsf).
%         - The preceding elements represent the hyperparameters for the covariance function (`covSE`).
%   x   - Input data matrix (T x N)
%         T: Number of observations or time steps
%         N: Number of features or dimensions in the data
%
% Outputs:
%   K  - Scaled Covariance Matrix (T x T)
%        The covariance matrix computed by scaling the base covariance (`K0`) with the scale factor `S`.
%   dK - Function handle to compute the derivative of the scaled covariance matrix with respect to hyperparameters
%        This handle accepts a matrix `Q` and returns the derivative `dhyp`.
%
% Description:
%   The `covScale` function computes a scaled covariance matrix based on the provided hyperparameters and input data.


    %% Extract Hyperparameters
    n = length(hyp);          % Total number of hyperparameters
    lsf = hyp(n);             % Log-scale factor (last element)
    hyp_cov = hyp(1:n-1);     % Covariance hyperparameters (all but last)
    hyp_cov = hyp_cov(:);     % Ensure `hyp_cov` is a column vector

    %% Compute Base Covariance and Its Derivative
    [K0, dK0] = covSE(hyp_cov, x);  % Compute base covariance and its derivative

    %% Compute Scale Factor
    sfx = exp(lsf);                  % Scale factor (exponential of log-scale)
    S = sfx * sfx;                    % Scaling matrix (outer product)

    %% Compute Scaled Covariance Matrix
    K = S .* K0;                      % Element-wise scaling of the covariance matrix

    %% Define Derivative Function Handle
    dK = @(Q) dirder(Q, S, K0, dK0, sfx);  % Function handle for derivative computation

end

%% Sub-function: dirder
function [dhyp] = dirder(Q, S, K0, dK0, sfx)
% DIRDER Computes the Derivative of the Scaled Covariance Matrix
%
% Syntax:
%   dhyp = dirder(Q, S, K0, dK0, sfx)
%
% Inputs:
%   Q    - Matrix for derivative computation (T x T)
%   S    - Scaling matrix (T x T)
%   K0   - Base covariance matrix (T x T)
%   dK0  - Function handle to compute derivative of `K0` with respect to hyperparameters
%   sfx  - Scale factor (scalar)
%
% Outputs:
%   dhyp - Derivative of the scaled covariance matrix with respect to hyperparameters (vector)
%
% Description:
%   The `dirder` function computes the derivative of the scaled covariance matrix `K`
%   with respect to the hyperparameters. 

    %% Compute Derivative with Respect to Base Covariance Hyperparameters
    dhyp0 = dK0(Q .* S);              % Derivative from base covariance

    %% Compute Derivative with Respect to Scale Factor
    Q_scaled = Q .* K0;                % Element-wise product
    qx = sum(Q_scaled * sfx, 2);       % Sum across columns after scaling
    dW = 2 * sfx * sum(qx);            % Derivative with respect to scale factor

    %% Combine Derivatives
    dhyp = [dhyp0; dW];                 % Concatenate derivatives

end
