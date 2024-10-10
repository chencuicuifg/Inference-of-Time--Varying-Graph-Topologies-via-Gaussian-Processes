function [K, dK, D2] = covSE(hyper, x)
% COVSE Squared Exponential Covariance Function with Unit Amplitude
%
% Syntax:
%   [K, dK, D2] = covSE(hyper, x)
%
% Inputs:
%   hyper - Hyperparameters vector
%           A numeric vector containing the hyperparameters for the covariance function.
%           Typically, this includes length-scale parameters for each feature.
%   x     - Input data matrix (T x N)
%           T: Number of observations or time steps
%           N: Number of features or dimensions in the data
%
% Outputs:
%   K  - Squared Exponential Covariance Matrix (T x T)
%        The covariance matrix computed using the Squared Exponential (SE) kernel.
%   dK - Function handle for the derivative of the covariance matrix with respect to hyperparameters
%        This handle accepts a matrix `Q` and returns the derivative vector `dhyper`.
%   D2 - Squared Euclidean Distance Matrix (T x T)
%        The matrix containing squared distances between each pair of observations.
%
% Description:
%   The `covSE` function computes the Squared Exponential (SE) covariance matrix for the given input data and hyperparameters.
%   The SE kernel is defined as:
%
%     k(x_i, x_j) = exp(-||x_i - x_j||^2 / 2)
%
%   where ||x_i - x_j||^2 is the squared Euclidean distance between observations `x_i` and `x_j`.
%   The function also returns a derivative handle `dK` for computing gradients with respect to the hyperparameters,
%   which is essential for gradient-based optimization methods.
%
%   Internally, the function leverages the `covMaha` function to compute the covariance matrix and its derivatives based
%   on the Mahalanobis distance.
%
% See Also:
%   covMaha

    
    %% Define Squared Exponential Kernel and Its Derivative
    k = @(d2) exp(-d2 / 2);             % Squared Exponential covariance function
    dk = @(d2, k_val) (-1/2) * k_val;   % Derivative of SE covariance function with respect to d2
    
    %% Compute Covariance Matrix and Its Derivative
    [K, dK, D2] = covMaha(k, dk, hyper, x);
    
end
